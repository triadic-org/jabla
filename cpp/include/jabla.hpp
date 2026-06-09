#pragma once
#include <vector>
#include <cblas.h>

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

inline int createTensor(std::vector<float> vector) {
  int idx = tensors.size();
  tensors.push_back(std::move(vector));
  return idx;
}

inline std::vector<float> getTensor(int idx) {
  return tensors.at(idx);
}

inline void clearTensors() {
  tensors.clear();
}

// matmul kernel: op(a) (m x k) . op(b) (k x n) -> (m x n) via cblas_sgemm, where
// op(x) is x or its transpose per transA/transB (CblasTrans reads the operand
// transposed in place -- no copy; this is what the matmul vjp uses for dY.Bt and
// At.dY). Reads the two registry buffers, writes the result as a NEW registry
// tensor, returns its id. m, n, k are the OPERATION dims (post-transpose). The
// row-major leading dim is the stored column count, so it depends on the flag:
// lda = transA ? m : k, ldb = transB ? k : n, ldc = n.
inline int matmul(int aId, int bId, int m, int n, int k, bool transA, bool transB) {
  // TODO: validate a and b exist and have correct dimensions
  std::vector<float> c(m * n, 0.0f);

  // Leading dim is column count for CblasRowMajor
  int lda = transA ? m : k;
  int ldb = transB ? k : n;

  cblas_sgemm(CblasRowMajor,
              (transA ? CblasTrans : CblasNoTrans),
              (transB ? CblasTrans : CblasNoTrans),
              m, n, k, 1.0f,
              tensors.at(aId).data(), lda,
              tensors.at(bId).data(), ldb, 0.0f,
              c.data(), n);

  return createTensor(std::move(c));
}

// add kernel: elementwise a + b, same shape. Reads the two registry buffers
// (assumed equal length -- a shape/length check is a later pass), writes the sum
// as a NEW registry tensor, returns its id. Elementwise, so no BLAS -- just a
// loop. Broadcasting comes later.
inline int add(int aId, int bId) {
  const std::vector<float>& a = tensors.at(aId);
  const std::vector<float>& b = tensors.at(bId);
  // TODO: validate a and b exist and have correct dimensions
 
  std::vector<float> c(a.size());
  for(std::size_t i = 0; i < c.size(); ++i) c[i] = a[i] + b[i];
  
  return createTensor(std::move(c));
}

} // namespace jabla
