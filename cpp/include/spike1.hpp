#pragma once
#include <vector>

// Spike buffers. Designed to be #included from one place as one translation unit.
// File-static state is a single instance - .at() during bring-up.
static std::vector<float> g_a, g_b, g_c;
static int g_m, g_k, g_n;

void mm_set_a(int i, double v) { g_a.at(i) = (float)v; } // <- mm_set_a
void mm_set_b(int i, double v) { g_b.at(i) = (float)v; } // <- mm_set_b
double mm_get_c(int i) { return (double)g_c.at(i); } // <- mm_get_c (getter!)

void mm_dims(int m, int k, int n) {
  g_m = m; g_k = k; g_n = n;
  g_a.assign(m*k, 0.0f);
}

void mm_run() {
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
