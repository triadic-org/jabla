#pragma once
#include <vector>

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
