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

TEST_CASE("createTensor stores data, getTensor returns it unchanged") {
  clearTensors();
  int a = createTensor({1.0f, 2.0f, 3.0f, 4.0f});
  auto v = getTensor(a);
  CHECK(v.size() == 4);
  CHECK(v[0] == 1.0f);
  CHECK(v[3] == 4.0f);
}

TEST_CASE("distinct tensors get distinct ids") {
  clearTensors();
  int a = createTensor({1.0f, 2.0f});
  int b = createTensor({5.0f, 6.0f, 7.0f});
  CHECK(a != b);
  CHECK(getTensor(b).size() == 3);
}

TEST_CASE("clear empties the registry; ids restart") {
  clearTensors();
  CHECK(tensors.empty());
  CHECK(createTensor({9.0f}) == 0);
}

// --- matmul (links OpenBLAS via cblas_sgemm) --------------------------------
// Contract under test: matmul(aIdx, bIdx, m, n, k) multiplies registry buffers
// a (m x k) and b (k x n) via cblas_sgemm, stores the m x n result as a NEW
// registry tensor, and returns its id (the result stays native -- no marshalling
// out). Param order is cblas's M, N, K. Oracle is the hand-computed 2x3 . 3x2
// also used by the jank matmul test.

TEST_CASE("matmul: (2x3)(3x2) = (2x2)") {
  clearTensors();
  int a = createTensor({1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f});      // 2x3, row-major
  int b = createTensor({7.0f,  8.0f,
                        9.0f,  10.0f,
                        11.0f, 12.0f});          // 3x2, row-major
  int c = matmul(a, b, 2, 2, 3);                 // m=2, n=2, k=3 (cblas M,N,K)
  auto v = getTensor(c);                         // matmul returns the result id

  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(58.0f));    // [0,0] = 1*7 + 2*9  + 3*11
  CHECK(v[1] == doctest::Approx(64.0f));    // [0,1] = 1*8 + 2*10 + 3*12
  CHECK(v[2] == doctest::Approx(139.0f));   // [1,0] = 4*7 + 5*9  + 6*11
  CHECK(v[3] == doctest::Approx(154.0f));   // [1,1] = 4*8 + 5*10 + 6*12
}

TEST_CASE("matmul leaves its inputs untouched (no aliasing)") {
  clearTensors();
  int a = createTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  int b = createTensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  matmul(a, b, 2, 2, 3);
  CHECK(getTensor(a) == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  CHECK(getTensor(b) == std::vector<float>{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
}

// --- add (elementwise; no BLAS) ---------------------------------------------
// Contract under test: add(aIdx, bIdx) sums the two registry buffers elementwise
// (same length), stores the result as a NEW registry tensor, returns its id.
// RED until you implement add in jabla.hpp.

TEST_CASE("add: elementwise (2x2)+(2x2)") {
  clearTensors();
  int a = createTensor({1.0f, 2.0f, 3.0f, 4.0f});
  int b = createTensor({10.0f, 20.0f, 30.0f, 40.0f});
  int c = add(a, b);
  auto v = getTensor(c);

  REQUIRE(v.size() == 4);
  CHECK(v[0] == doctest::Approx(11.0f));
  CHECK(v[1] == doctest::Approx(22.0f));
  CHECK(v[2] == doctest::Approx(33.0f));
  CHECK(v[3] == doctest::Approx(44.0f));
}

TEST_CASE("add leaves its inputs untouched (no aliasing)") {
  clearTensors();
  int a = createTensor({1.0f, 2.0f, 3.0f, 4.0f});
  int b = createTensor({10.0f, 20.0f, 30.0f, 40.0f});
  add(a, b);
  CHECK(getTensor(a) == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  CHECK(getTensor(b) == std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f});
}
