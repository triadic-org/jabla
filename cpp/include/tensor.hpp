#pragma once
#include <vector>
#include <cblas.h>

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

// (defn matmul
//   "a (m x k) . b (k x n) -> (m x n). Forward routes through blas/sgemm; records a
//   node whose vjp is two matmuls of the upstream gradient with a and b."
//   [t1 t2]
//   (let [[m k] (get-shape t1)
//         [_ n] (get-shape t2)
//         t3-id (cpp/matmul (:id t1) (:id t2) m n k)]
//     {:shape [m n] :dtype DEFAULT_DTYPE :id t3-id}))
