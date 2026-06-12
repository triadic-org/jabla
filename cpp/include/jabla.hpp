#pragma once
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <vector>

// Native backend for jabla.tensor: the buffer registry plus the ops (matmul,
// ...). One header for now; split (jabla_nn, jabla_cuda, ...) once the
// organization is clearer.
//
// Everything lives in `namespace jabla` so the C++ symbols can keep their
// natural names even when a jank op shares them. jank calls these qualified --
// (cpp/jabla.matmul ...) -> jabla::matmul(...) -- and a qualified call can't be
// shadowed by the same-named unqualified jank var in the generated TU. That's
// what lets (defn matmul ...) call (cpp/jabla.matmul ...) without collision.
// See docs/jank-notes.md (naming across C++/jank).
namespace jabla {

  // Registry: tensors own contiguous float32 buffers; jank refers to them by id
  // (the index here). Bulk data never crosses into jank.
  inline std::vector<std::vector<float>> tensors;

  inline int create_tensor(std::vector<float> vector) {
    int idx = tensors.size();
    tensors.push_back(std::move(vector));
    return idx;
  }

  inline std::vector<float> get_tensor(int idx) {
    return tensors.at(idx);
  }

  inline void clear_tensors() {
    tensors.clear();
  }

  // matmul kernel: op(a) (m x k) . op(b) (k x n) -> (m x n) via cblas_sgemm, where
  // op(x) is x or its transpose per trans_a/trans_b (CblasTrans reads the operand
  // transposed in place -- no copy; this is what the matmul vjp uses for dY.Bt and
  // At.dY). Reads the two registry buffers, writes the result as a NEW registry
  // tensor, returns its id. m, n, k are the OPERATION dims (post-transpose). The
  // row-major leading dim is the stored column count, so it depends on the flag:
  // lda = trans_a ? m : k, ldb = trans_b ? k : n, ldc = n.
  inline int matmul(int a_id, int b_id, int m, int n, int k, bool trans_a, bool trans_b) {
    // TODO: validate a and b exist and have correct dimensions
    std::vector<float> c(m * n, 0.0f);

    // Leading dim is column count for CblasRowMajor
    int lda = trans_a ? m : k;
    int ldb = trans_b ? k : n;

    cblas_sgemm(CblasRowMajor,
                (trans_a ? CblasTrans : CblasNoTrans),
                (trans_b ? CblasTrans : CblasNoTrans),
                m, n, k, 1.0f,
                tensors.at(a_id).data(), lda,
                tensors.at(b_id).data(), ldb, 0.0f,
                c.data(), n);

    return create_tensor(std::move(c));
  }

  // add kernel: elementwise a + b, same shape. Reads the two registry buffers
  // (assumed equal length -- a shape/length check is a later pass), writes the sum
  // as a NEW registry tensor, returns its id. Elementwise, so no BLAS -- just a
  // loop. Broadcasting comes later.
  inline int add(int a_id, int b_id) {
    const std::vector<float>& a = tensors.at(a_id);
    const std::vector<float>& b = tensors.at(b_id);
    // TODO: validate a and b exist and have correct dimensions
 
    std::vector<float> c(a.size());
    for(std::size_t i = 0; i < c.size(); ++i) c[i] = a[i] + b[i];

    return create_tensor(std::move(c));
  }

  // mul kernel: elementwise a * b (Hadamard product), same shape. Like add, no BLAS.
  inline int mul(int a_id, int b_id) {
    const std::vector<float>& a = tensors.at(a_id);
    const std::vector<float>& b = tensors.at(b_id);
    std::vector<float> c(a.size());
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = a[i] * b[i];
    return create_tensor(std::move(c));
  }

  inline int relu(int x_id) {
    const std::vector<float>& x = tensors.at(x_id);
    std::vector<float> y(x.size());

    for (std::size_t i= 0; i < y.size(); ++i) {
      float xi = x[i];
      y[i] = xi > 0.0f? xi : 0.0f;
    }
    return create_tensor(std::move(y));
  }

  inline int relu_backward(int x_id, int dy_id) {
    const std::vector<float>& x = tensors.at(x_id);
    const std::vector<float>& dy = tensors.at(dy_id);
    std::vector<float> dx(x.size());

    for (std::size_t i = 0; i < dx.size(); ++i){
      float xi = x[i];
      dx[i] = xi > 0.0f ? dy[i] : 0.0f;
    }
    return create_tensor(std::move(dx));
  }

  // gelu kernel: elementwise GELU activation, tanh approximation (matches nanoGPT's
  // new_gelu and llm.c): gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
  // (Exact-erf GELU is an alternative -- switch the body if validating vs nn.GELU().)
  inline int gelu(int x_id) {
    const std::vector<float>& x = tensors.at(x_id);

    std::vector<float> y(x.size());
    const float s = 0.7978845608028654f;  // sqrt(2/pi)

    for (std::size_t i = 0; i < y.size(); ++i) {
      float xi = x[i];
      float inner = s * (xi + 0.044715f * xi * xi * xi);
      y[i] = 0.5f * xi * (1.0f + std::tanh(inner));
    }
    return create_tensor(std::move(y));
  }

  // gelu_backward: the fused gelu vjp -- dx = dy * gelu'(x), elementwise. Derivative of
  // the tanh approximation (s = sqrt(2/pi), inner = s (x + 0.044715 x^3)):
  //   gelu'(x) = 0.5 (1 + tanh(inner)) + 0.5 x (1 - tanh^2(inner)) s (1 + 0.134145 x^2)
  inline int gelu_backward(int x_id, int dy_id) {
    const std::vector<float>& x = tensors.at(x_id);
    const std::vector<float>& dy = tensors.at(dy_id);

    std::vector<float> dx(x.size());
    const float s = 0.7978845608028654f;  // sqrt(2/pi)

    for (std::size_t i = 0; i < dx.size(); ++i) {
      float xi = x[i];
      float inner = s * (xi + 0.044715f * xi * xi * xi);
      float t = std::tanh(inner);
      float dinner = s * (1.0f + 0.134145f * xi * xi);  // 0.134145 = 3 * 0.044715
      float dgelu = 0.5f * (1.0f + t) + 0.5f * xi * (1.0f - t * t) * dinner;
      dx[i] = dy[i] * dgelu;
    }
    return create_tensor(std::move(dx));
  }

  inline int softmax(int x_id, int rows, int cols) {
    const std::vector<float>& x = tensors.at(x_id);
    std::vector<float> y(x.size());

    for (int r = 0; r < rows; ++r) {
      const float* xr = x.data() + r * cols;
      float* yr = y.data() + r * cols;

      float row_max = *std::max_element(xr, xr + cols); 
      float row_sum = 0.0f; 

      // Subtract row max from exp(value), sum results
      for (int c = 0; c < cols; ++c) {
        yr[c] = std::exp(xr[c] - row_max);
        row_sum += yr[c];
      }
      for (int c = 0; c < cols; ++c)  yr[c] /= row_sum;
    }
    return create_tensor(std::move(y));
  }

  // s_id = softmax tensor
  // dy_id = grad out tensor
  inline int softmax_backward(int s_id, int dy_id, int rows, int cols) {
    const std::vector<float>& s = tensors.at(s_id);
    const std::vector<float>& dy = tensors.at(dy_id);

    std::vector<float> dx(s.size());

    // Compute rowdot
    for (int r = 0; r < rows; ++r) {
      const float* sr = s.data() + r * cols;
      const float* dyr = dy.data() + r * cols;
      float* dxr = dx.data() + r * cols;
      float dot = 0.0f;

      for (int c = 0; c < cols; ++c) dot += sr[c] * dyr[c];
      for (int c = 0; c < cols; ++c) dxr[c] = sr[c] * (dyr[c] - dot);
    }
    return create_tensor(std::move(dx));
  }

} // namespace jabla
