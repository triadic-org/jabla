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
  tensors.push_back(vector);
  return idx;
}

inline std::vector<float> getTensor(int idx) {
  return tensors.at(idx);
}

inline void clearTensors() {
  tensors.clear();
}

// matmul kernel: a (m x k) . b (k x n) -> (m x n) via cblas_sgemm. Reads the two
// registry buffers, writes the result as a NEW registry tensor, returns its id.
// Param order is cblas's M, N, K. Row-major leading dims: lda=k, ldb=n, ldc=n.
inline int matmul(int aId, int bId, int m, int n, int k) {
  std::vector<float> c(m * n, 0.0f);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k, 1.0f,
              tensors.at(aId).data(), k,
              tensors.at(bId).data(), n, 0.0f,
              c.data(), n);

  tensors.push_back(std::move(c));
  return (int)tensors.size() - 1;
}

// add kernel (scaffold -- body is yours): elementwise a + b, same shape. Reads
// the two registry buffers (they should be equal length -- consider asserting
// it), writes the sum as a NEW registry tensor, returns its id. Elementwise, so
// no BLAS -- just a loop. Broadcasting comes later. Mirror matmul's tail:
// push_back(std::move(result)); return (int)tensors.size() - 1;
inline int add(int aId, int bId) {
  return -1; // TODO(you): sum tensors.at(aId) + tensors.at(bId) into a new tensor
}

} // namespace jabla
