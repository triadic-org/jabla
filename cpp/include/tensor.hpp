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

