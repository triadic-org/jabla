// C++ unit tests for the tensor buffer registry (cpp/include/tensor.hpp), using
// doctest (vendored single header). Run locally, no jank: `make cpp-test`.
// Catches C++ logic bugs (the class that cost a devbox round-trip in the BLAS
// spike) in the fast clang loop.
//
// RED until you implement the bodies in tensor.hpp.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.hpp"

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

// TODO(you): once matmul lands in tensor.hpp, add a 2x3 . 3x2 ==
// [[58 64] [139 154]] case (same oracle as blas), linked vs OpenBLAS.
