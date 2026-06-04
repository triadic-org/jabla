#pragma once
#include <vector>
#include <cblas.h>

// Spike buffers. Designed to be #included from one place as one translation unit.
inline std::vector<float> g_a, g_b, g_c;
inline int g_m, g_k, g_n;

// a is m x k, b is k x n; returns c at m x n.
inline void mm_dims(int m, int k, int n) {
  g_m = m; g_k = k; g_n = n;
  g_a.assign(m*k, 0.0f);
  g_b.assign(k*n, 0.0f);
  g_c.assign(m*n, 0.0f);
}

inline void mm_set_a(int i, double v) { g_a.at(i) = (float)v; }
inline void mm_set_b(int i, double v) { g_b.at(i) = (float)v; }
inline double mm_get_c(int i) { return (double)g_c.at(i); }

inline void mm_run() {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, cBlasNoTrans,
              g_m, g_n, g_k, 1.0f,
              g_a.data(), g_k,
              g_b.data(), g_n, 0.0f,
              g_c.data(), g_n);
}
