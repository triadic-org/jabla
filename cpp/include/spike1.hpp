#pragma once
#include <vector>

// Spike buffers. Designed to be #included from one place as one translation unit.
inline std::vector<float> g_a, g_b, g_c;
inline int g_m, g_k, g_n;

inline void mm_dims(int m, int k, int n) {
  g_m = m; g_k = k; g_n = n;
  g_a.assign(m*k, 0.0f);
}

inline void mm_set_a(int i, double v) { g_a.at(i) = (float)v; } // <- mm_set_a
inline void mm_set_b(int i, double v) { g_b.at(i) = (float)v; } // <- mm_set_b
inline double mm_get_c(int i) { return (double)g_c.at(i); } // <- mm_get_c (getter!)

inline void mm_run() {
  for (int i = 0; i < g_m; ++i) {
    for (int j = 0; j < g_n; ++j) {
      float s = 0.0f;
      for (int p = 0; p < g_k; ++p) {
        s += g_a.at(i * g_k + p) * g_b.at(p * g_n + j);
        g_c.at(i * g_n + j) = s;
      }
    }
  }
}
