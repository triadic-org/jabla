// C++ unit tests for the tensor backend (cpp/include/jabla.hpp): the buffer
// registry plus the ops (matmul, ...). Uses doctest (vendored single header).
// Run locally, no jank: `make cpp-test`. Catches C++ logic bugs (the class that
// cost a devbox round-trip in the BLAS spike) in the fast clang loop.
//
// The backend lives in namespace jabla; `using` it here keeps the calls terse.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "jabla.hpp"

using namespace jabla;

TEST_CASE("create_tensor stores data, get_tensor returns it unchanged") {
  clear_tensors();
  int a = create_tensor({1.0f, 2.0f, 3.0f, 4.0f});
  auto v = get_tensor(a);
  CHECK(v.size() == 4);
  CHECK(v[0] == 1.0f);
  CHECK(v[3] == 4.0f);
}

TEST_CASE("distinct tensors get distinct ids") {
  clear_tensors();
  int a = create_tensor({1.0f, 2.0f});
  int b = create_tensor({5.0f, 6.0f, 7.0f});
  CHECK(a != b);
  CHECK(get_tensor(b).size() == 3);
}

TEST_CASE("clear empties the registry; ids restart") {
  clear_tensors();
  CHECK(tensors.empty());
  CHECK(create_tensor({9.0f}) == 0);
}

// --- matmul (links OpenBLAS via cblas_sgemm) --------------------------------
// Contract under test: matmul(a_id, b_id, m, n, k, trans_a, trans_b) multiplies
// op(a) (m x k) . op(b) (k x n) via cblas_sgemm -- op(x) is x or its transpose per
// the flags -- stores the m x n result as a NEW registry tensor, returns its id
// (stays native, no marshalling out). m, n, k are the OPERATION dims (post-transpose).
// The no-trans oracle is the hand-computed 2x3 . 3x2 also used by the jank test; the
// transposed cases below exercise the leading-dim math the matmul vjp relies on.

TEST_CASE("matmul: (2x3)(3x2) = (2x2), no transpose") {
  clear_tensors();
  int a = create_tensor({1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f});      // 2x3, row-major
  int b = create_tensor({7.0f,  8.0f,
                        9.0f,  10.0f,
                        11.0f, 12.0f});          // 3x2, row-major
  int c = matmul(a, b, 2, 2, 3, false, false);   // m=2, n=2, k=3 (cblas M,N,K)
  auto v = get_tensor(c);                         // matmul returns the result id

  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(58.0f));    // [0,0] = 1*7 + 2*9  + 3*11
  CHECK(v[1] == doctest::Approx(64.0f));    // [0,1] = 1*8 + 2*10 + 3*12
  CHECK(v[2] == doctest::Approx(139.0f));   // [1,0] = 4*7 + 5*9  + 6*11
  CHECK(v[3] == doctest::Approx(154.0f));   // [1,1] = 4*8 + 5*10 + 6*12
}

// trans_b: the dA = dY . Bt shape from the matmul vjp. With Y = A.B for A(2x3) B(3x4),
// dY is 2x4 and dA = dY . Bt is (2x4)(4x3) = 2x3. Here dY plays "a" (no trans), B
// plays "b" with trans_b=true; m=2, n=3, k=4. Oracle: dY . (B transposed by hand).
TEST_CASE("matmul: trans_b reads b transposed in place (dY . Bt)") {
  clear_tensors();
  int dy = create_tensor({1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 6.0f, 7.0f, 8.0f});         // 2x4
  int b  = create_tensor({1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 6.0f, 7.0f, 8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f});      // 3x4 (so Bt is 4x3)
  int c = matmul(dy, b, 2, 3, 4, false, true);            // (2x4)(4x3) = 2x3
  auto v = get_tensor(c);

  REQUIRE(v.size() == 6);
  // row 0 of dY . Bt: dot([1,2,3,4], each row of B)
  CHECK(v[0] == doctest::Approx(30.0f));    // [0,0] = 1*1+2*2+3*3+4*4
  CHECK(v[1] == doctest::Approx(70.0f));    // [0,1] = 1*5+2*6+3*7+4*8
  CHECK(v[2] == doctest::Approx(110.0f));   // [0,2] = 1*9+2*10+3*11+4*12
  CHECK(v[5] == doctest::Approx(278.0f));   // [1,2] = 5*9+6*10+7*11+8*12
}

// trans_a: the dB = At . dY shape from the matmul vjp. For A(2x3) B(3x4), dB = At . dY
// is (3x2)(2x4) = 3x4. Here A plays "a" with trans_a=true, dY plays "b"; m=3, n=4, k=2.
TEST_CASE("matmul: trans_a reads a transposed in place (At . dY)") {
  clear_tensors();
  int a  = create_tensor({1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f});               // 2x3 (so At is 3x2)
  int dy = create_tensor({1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 6.0f, 7.0f, 8.0f});         // 2x4
  int c = matmul(a, dy, 3, 4, 2, true, false);            // (3x2)(2x4) = 3x4
  auto v = get_tensor(c);

  REQUIRE(v.size() == 12);
  // At col j is A col j; row 0 of At is A's col 0 = [1,4]; dot with dY cols
  CHECK(v[0] == doctest::Approx(21.0f));     // [0,0] = 1*1 + 4*5
  CHECK(v[3] == doctest::Approx(36.0f));     // [0,3] = 1*4 + 4*8
  CHECK(v[11] == doctest::Approx(60.0f));    // [2,3] = 3*4 + 6*8
}

TEST_CASE("matmul leaves its inputs untouched (no aliasing)") {
  clear_tensors();
  int a = create_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  int b = create_tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  matmul(a, b, 2, 2, 3, false, false);
  CHECK(get_tensor(a) == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  CHECK(get_tensor(b) == std::vector<float>{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
}

// --- add (elementwise; no BLAS) ---------------------------------------------
// Contract under test: add(a_id, b_id) sums the two registry buffers elementwise
// (same length), stores the result as a NEW registry tensor, returns its id.
// RED until you implement add in jabla.hpp.

TEST_CASE("add: elementwise (2x2)+(2x2)") {
  clear_tensors();
  int a = create_tensor({1.0f, 2.0f, 3.0f, 4.0f});
  int b = create_tensor({10.0f, 20.0f, 30.0f, 40.0f});
  int c = add(a, b);
  auto v = get_tensor(c);

  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(11.0f));
  CHECK(v[1] == doctest::Approx(22.0f));
  CHECK(v[2] == doctest::Approx(33.0f));
  CHECK(v[3] == doctest::Approx(44.0f));
}

TEST_CASE("add leaves its inputs untouched (no aliasing)") {
  clear_tensors();
  int a = create_tensor({1.0f, 2.0f, 3.0f, 4.0f});
  int b = create_tensor({10.0f, 20.0f, 30.0f, 40.0f});
  add(a, b);
  CHECK(get_tensor(a) == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  CHECK(get_tensor(b) == std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f});
}

// --- mul (elementwise Hadamard; no BLAS) ------------------------------------
TEST_CASE("mul: elementwise (2x2) * (2x2)") {
  clear_tensors();
  int a = create_tensor({1.0f, 2.0f, 3.0f, 4.0f});
  int b = create_tensor({5.0f, 6.0f, 7.0f, 8.0f});
  auto v = get_tensor(mul(a, b));
  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(5.0f));     // 1*5
  CHECK(v[1] == doctest::Approx(12.0f));    // 2*6
  CHECK(v[3] == doctest::Approx(32.0f));    // 4*8
}

// --- gelu (tanh approximation) + its fused backward -------------------------
// Reference values: PyTorch F.gelu(., approximate='tanh'): gelu(1)~0.8412,
// gelu(-1)~-0.1588, gelu(0)=0. gelu'(0) = 0.5 (so gelu_backward(0, dy) = 0.5*dy).
TEST_CASE("gelu: tanh-approx known values") {
  clear_tensors();
  int x = create_tensor({0.0f, 1.0f, -1.0f, 2.0f});
  auto v = get_tensor(gelu(x));
  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(0.0f));
  CHECK(v[1] == doctest::Approx(0.8412f).epsilon(0.01));
  CHECK(v[2] == doctest::Approx(-0.1588f).epsilon(0.01));
}

TEST_CASE("gelu_backward = dy * gelu'(x); gelu'(0) = 0.5") {
  clear_tensors();
  int x  = create_tensor({0.0f});
  int dy = create_tensor({2.0f});
  auto v = get_tensor(gelu_backward(x, dy));
  REQUIRE(v.size() == 1);
  CHECK(v[0] == doctest::Approx(1.0f));    // 2.0 * gelu'(0)=0.5
}

// --- relu (elementwise max(0, x)) + its backward ----------------------------
// Contract: relu(x_id) applies max(0, x) elementwise, stores a NEW registry tensor,
// returns its id. relu_backward(x_id, dy_id) is the vjp -- dx = dy * (x > 0), the
// subgradient with the kink mapped to 0 (relu'(0) := 0; the jank grad-check keeps
// zeros out of its inputs so this choice never shows up there).
TEST_CASE("relu: max(0, x) elementwise, including negatives and zero") {
  clear_tensors();
  int x = create_tensor({-1.0f, 2.0f, 0.0f, 3.0f});
  auto v = get_tensor(relu(x));
  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(0.0f));    // negative clamps to 0
  CHECK(v[1] == doctest::Approx(2.0f));    // positive passes through
  CHECK(v[2] == doctest::Approx(0.0f));    // zero stays 0
  CHECK(v[3] == doctest::Approx(3.0f));
}

TEST_CASE("relu leaves its input untouched (no aliasing)") {
  clear_tensors();
  int x = create_tensor({-1.0f, 2.0f, 0.0f, 3.0f});
  relu(x);
  CHECK(get_tensor(x) == std::vector<float>{-1.0f, 2.0f, 0.0f, 3.0f});
}

// Uncomment once relu_backward(x_id, dy_id) lands. dx = dy * (x > 0): the gradient
// passes through where x is positive and is killed where x is negative; at x = 0
// the subgradient is 0 (relu'(0) := 0).
TEST_CASE("relu_backward = dy * (x > 0); kink at 0 maps to 0") {
  clear_tensors();
  int x  = create_tensor({-1.0f, 2.0f, 0.0f, 3.0f});
  int dy = create_tensor({10.0f, 10.0f, 10.0f, 10.0f});
  auto v = get_tensor(relu_backward(x, dy));
  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(0.0f));     // x<0 -> grad killed
  CHECK(v[1] == doctest::Approx(10.0f));    // x>0 -> grad passes
  CHECK(v[2] == doctest::Approx(0.0f));     // x=0 -> relu'(0):=0
  CHECK(v[3] == doctest::Approx(10.0f));
}

// --- softmax (row-wise) + its coupled backward ------------------------------
// The first op whose backward is NOT elementwise: every output in a row depends on
// every input in that row, so the vjp is a row reduction, not a diagonal scale.
// Contract: softmax(x_id, rows, cols) treats the buffer as rows x cols row-major and
// applies softmax along each row (subtract the row max first for numerical stability,
// then exp / rowsum). softmax_backward(s_id, dy_id, rows, cols) takes the FORWARD OUTPUT
// s (not x) and the upstream dy, and returns dx_i = s_i (dy_i - sum_j dy_j s_j) --
// i.e. dx = s * (dy - rowdot), where rowdot is the per-row dot(dy, s) broadcast.
// (Signature shown as (id, rows, cols); adapt if you carry shape differently.)
//
// Uncomment once softmax + softmax_backward land in jabla.hpp. Reference values are
// PyTorch's softmax([1,2,3]) = [.0900, .2447, .6652].
TEST_CASE("softmax: row sums to 1; reference softmax([1,2,3])") {
  clear_tensors();
  int x = create_tensor({1.0f, 2.0f, 3.0f});       // 1 row x 3 cols
  auto v = get_tensor(softmax(x, 1, 3));
  REQUIRE(v.size() == 3);
  CHECK(v[0] + v[1] + v[2] == doctest::Approx(1.0f));   // normalized
  CHECK(v[0] == doctest::Approx(0.0900f).epsilon(0.01));
  CHECK(v[1] == doctest::Approx(0.2447f).epsilon(0.01));
  CHECK(v[2] == doctest::Approx(0.6652f).epsilon(0.01));
}

// Stability: softmax([1001,1002,1003]) must equal softmax([1,2,3]) -- the row-max
// subtraction is what prevents exp() overflow. This is the test that catches a
// kernel that forgot to subtract the max.
TEST_CASE("softmax: shift-invariant (row-max subtraction prevents overflow)") {
  clear_tensors();
  int x = create_tensor({1001.0f, 1002.0f, 1003.0f});
  auto v = get_tensor(softmax(x, 1, 3));
  REQUIRE(v.size() == 3);
  CHECK(v[0] == doctest::Approx(0.0900f).epsilon(0.01));
  CHECK(v[2] == doctest::Approx(0.6652f).epsilon(0.01));
}

// Two rows at once: confirms the reduction is PER ROW, not over the whole buffer.
TEST_CASE("softmax: independent per row") {
  clear_tensors();
  int x = create_tensor({1.0f, 2.0f, 3.0f,
                        3.0f, 2.0f, 1.0f});         // 2 rows x 3 cols
  auto v = get_tensor(softmax(x, 2, 3));
  REQUIRE(v.size() == 6);
  CHECK(v[0] + v[1] + v[2] == doctest::Approx(1.0f));
  CHECK(v[3] + v[4] + v[5] == doctest::Approx(1.0f));
  CHECK(v[5] == doctest::Approx(0.0900f).epsilon(0.01));  // row 1 is row 0 reversed
}
//
// // Backward oracle: s = softmax([1,2,3]) = [.0900, .2447, .6652], dy = [1, 0, 0].
// // rowdot = dot(dy, s) = .0900. dx = s * (dy - rowdot):
// //   dx0 = .0900 * (1 - .0900) =  .0819
// //   dx1 = .2447 * (0 - .0900) = -.0220
// //   dx2 = .6652 * (0 - .0900) = -.0599
// // (Note it takes s, the forward output, not x -- the kernel saves a recompute.)
TEST_CASE("softmax_backward = s * (dy - rowdot(dy, s))") {
  clear_tensors();
  int s  = create_tensor({0.0900f, 0.2447f, 0.6652f});  // a softmax row
  int dy = create_tensor({1.0f, 0.0f, 0.0f});
  auto v = get_tensor(softmax_backward(s, dy, 1, 3));
  REQUIRE(v.size() == 3);
  CHECK(v[0] == doctest::Approx(0.0819f).epsilon(0.02));
  CHECK(v[1] == doctest::Approx(-0.0220f).epsilon(0.02));
  CHECK(v[2] == doctest::Approx(-0.0599f).epsilon(0.02));
}
