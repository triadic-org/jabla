#pragma once
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <numeric>
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

  inline int create_tensor(std::vector<float> data) {
    int idx = tensors.size();
    tensors.push_back(std::move(data));
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
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = a[i] + b[i];

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

  // relu kernel: elementwise max(0, x). No BLAS -- a plain loop. The ternary
  // (x > 0 ? x : 0) lowers to the same maxss instruction as std::max but needs
  // no <algorithm>.
  inline int relu(int x_id) {
    const std::vector<float>& x = tensors.at(x_id);
    std::vector<float> y(x.size());

    for (std::size_t i = 0; i < y.size(); ++i) {
      float xi = x[i];
      y[i] = xi > 0.0f ? xi : 0.0f;
    }
    return create_tensor(std::move(y));
  }

  // relu_backward: the relu vjp -- dx = dy * (x > 0), elementwise. The subgradient
  // passes the upstream grad where x is positive and kills it where x <= 0
  // (relu'(0) := 0; the jank grad-check keeps zeros out of its inputs).
  inline int relu_backward(int x_id, int dy_id) {
    const std::vector<float>& x = tensors.at(x_id);
    const std::vector<float>& dy = tensors.at(dy_id);
    std::vector<float> dx(x.size());

    for (std::size_t i = 0; i < dx.size(); ++i) {
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

  // softmax kernel: row-wise softmax over a rows x cols row-major buffer. Two passes
  // per row for numerical stability: subtract the row max before exp (shift-invariant,
  // prevents overflow), then divide by the row sum. Writes a NEW registry tensor.
  inline int softmax(int x_id, int rows, int cols) {
    const std::vector<float>& x = tensors.at(x_id);
    std::vector<float> y(x.size());

    for (int r = 0; r < rows; ++r) {
      const float* xr = x.data() + r * cols;
      float* yr = y.data() + r * cols;

      float row_max = *std::max_element(xr, xr + cols);
      float row_sum = 0.0f;

      // exp(value - row max), accumulating the row sum in the same pass
      for (int c = 0; c < cols; ++c) {
        yr[c] = std::exp(xr[c] - row_max);
        row_sum += yr[c];
      }
      for (int c = 0; c < cols; ++c) yr[c] /= row_sum;
    }
    return create_tensor(std::move(y));
  }

  // softmax_backward: the coupled softmax vjp. Takes s = the FORWARD OUTPUT (not x)
  // and the upstream dy; per row dx = s * (dy - rowdot), where rowdot = dot(dy, s).
  // Every output in a row depends on every input, so the vjp is a row reduction.
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

  // layernorm kernel: per row, out = gamma*norm + beta where norm = (x-mu)/sqrt(var+eps),
  // mu/var are the population (1/N -- matches PyTorch) mean/variance of the row, eps=1e-5.
  // gamma/beta are per-COLUMN (length cols), shared across rows -- indexed by c, NOT
  // row-offset like x/y. Two stat passes (mean, then variance) + one output pass.
  inline int layernorm(int x_id, int gamma_id, int beta_id, int rows, int cols) {
    const std::vector<float>& x = tensors.at(x_id);
    const std::vector<float>& gamma = tensors.at(gamma_id);
    const std::vector<float>& beta = tensors.at(beta_id);
    const float eps = 1e-5f;
    std::vector<float> y(x.size());

    for (int r = 0; r < rows; ++r) {
      const float* xr = x.data() + r * cols;
      float* yr = y.data() + r * cols;

      float mu = std::accumulate(xr, xr + cols, 0.0f) / cols;
      float var = 0.0f;
      for (int c = 0; c < cols; ++c) { float d = xr[c] - mu; var += d * d; }
      var /= cols;
      float rstd = 1.0f / std::sqrt(var + eps);

      for (int c = 0; c < cols; ++c)
        yr[c] = gamma[c] * ((xr[c] - mu) * rstd) + beta[c];
    }
    return create_tensor(std::move(y));
  }

  // layernorm_backward: the coupled x-gradient. Recomputes mu/rstd/norm from x. With
  // g = gamma*dy (the affine folded into the upstream grad), per row:
  //   dx = rstd * (g - mean(g) - norm * mean(g*norm)).
  inline int layernorm_backward(int x_id, int gamma_id, int dy_id, int rows, int cols) {
    const std::vector<float>& x = tensors.at(x_id);
    const std::vector<float>& gamma = tensors.at(gamma_id);
    const std::vector<float>& dy = tensors.at(dy_id);
    const float eps = 1e-5f;
    std::vector<float> dx(x.size());

    for (int r = 0; r < rows; ++r) {
      const float* xr = x.data() + r * cols;
      const float* dyr = dy.data() + r * cols;
      float* dxr = dx.data() + r * cols;

      float mu = std::accumulate(xr, xr + cols, 0.0f) / cols;
      float var = 0.0f;
      for (int c = 0; c < cols; ++c) { float d = xr[c] - mu; var += d * d; }
      var /= cols;
      float rstd = 1.0f / std::sqrt(var + eps);

      float g_mean = 0.0f, gn_mean = 0.0f;
      for (int c = 0; c < cols; ++c) {
        float norm = (xr[c] - mu) * rstd;
        float g = gamma[c] * dyr[c];
        g_mean += g;
        gn_mean += g * norm;
      }
      g_mean /= cols;
      gn_mean /= cols;

      for (int c = 0; c < cols; ++c) {
        float norm = (xr[c] - mu) * rstd;
        float g = gamma[c] * dyr[c];
        dxr[c] = rstd * (g - g_mean - norm * gn_mean);
      }
    }
    return create_tensor(std::move(dx));
  }

  // layernorm_gamma_grad: dgamma[c] = sum over rows of dy * norm (recomputes norm from
  // x). Length cols -- a column reduction (axis 0).
  inline int layernorm_gamma_grad(int x_id, int dy_id, int rows, int cols) {
    const std::vector<float>& x = tensors.at(x_id);
    const std::vector<float>& dy = tensors.at(dy_id);
    const float eps = 1e-5f;
    std::vector<float> dgamma(cols, 0.0f);

    for (int r = 0; r < rows; ++r) {
      const float* xr = x.data() + r * cols;
      const float* dyr = dy.data() + r * cols;

      float mu = std::accumulate(xr, xr + cols, 0.0f) / cols;
      float var = 0.0f;
      for (int c = 0; c < cols; ++c) { float d = xr[c] - mu; var += d * d; }
      var /= cols;
      float rstd = 1.0f / std::sqrt(var + eps);

      for (int c = 0; c < cols; ++c) dgamma[c] += dyr[c] * ((xr[c] - mu) * rstd);
    }
    return create_tensor(std::move(dgamma));
  }

  // layernorm_beta_grad: dbeta[c] = sum over rows of dy. Length cols.
  inline int layernorm_beta_grad(int dy_id, int rows, int cols) {
    const std::vector<float>& dy = tensors.at(dy_id);
    std::vector<float> dbeta(cols, 0.0f);

    for (int r = 0; r < rows; ++r) {
      const float* dyr = dy.data() + r * cols;
      for (int c = 0; c < cols; ++c) dbeta[c] += dyr[c];
    }
    return create_tensor(std::move(dbeta));
  }

  // cross_entropy kernel: mean over rows of -log(softmax(logits_r)[targets_r]). The
  // softmax is fused (subtract row max) for stability. `targets` is one class index
  // per row, passed as float and cast to int (reuses the create_tensor marshaling
  // path). Returns a 1-element registry tensor (the scalar loss).
  inline int cross_entropy(int logits_id, std::vector<float> targets, int rows, int cols) {
    const std::vector<float>& logits = tensors.at(logits_id);
    float loss = 0.0f;

    for (int r = 0; r < rows; ++r) {
      const float* lr = logits.data() + r * cols;
      float row_max = *std::max_element(lr, lr + cols);
      float sum = 0.0f;
      for (int c = 0; c < cols; ++c) sum += std::exp(lr[c] - row_max);
      int t = static_cast<int>(targets[r]);
      // -log(softmax[t]) = log(sum) - (logit_t - row_max)
      loss += std::log(sum) - (lr[t] - row_max);
    }
    return create_tensor(std::vector<float>{loss / rows});
  }

  // cross_entropy_backward: dlogits = (softmax(logits) - onehot(targets)) / rows -- the
  // clean fused gradient. Assumes the loss is the backward ROOT (seeded with 1), the
  // only way a scalar loss is used, so it does not scale by an upstream grad.
  inline int cross_entropy_backward(int logits_id, std::vector<float> targets, int rows, int cols) {
    const std::vector<float>& logits = tensors.at(logits_id);
    std::vector<float> dx(logits.size());

    for (int r = 0; r < rows; ++r) {
      const float* lr = logits.data() + r * cols;
      float* dr = dx.data() + r * cols;
      float row_max = *std::max_element(lr, lr + cols);
      float sum = 0.0f;
      for (int c = 0; c < cols; ++c) { dr[c] = std::exp(lr[c] - row_max); sum += dr[c]; }
      for (int c = 0; c < cols; ++c) dr[c] /= sum;          // dr = softmax
      int t = static_cast<int>(targets[r]);
      dr[t] -= 1.0f;                                         // - onehot
      for (int c = 0; c < cols; ++c) dr[c] /= rows;         // / N (mean)
    }
    return create_tensor(std::move(dx));
  }

  // embedding kernel: gather rows of W (vocab x dim) by integer index. out[i] =
  // W[indices[i]], shape (len(indices) x dim). indices passed as float, cast to int.
  inline int embedding(int w_id, std::vector<float> indices, int dim) {
    const std::vector<float>& w = tensors.at(w_id);
    int n = static_cast<int>(indices.size());
    std::vector<float> out(n * dim);

    for (int i = 0; i < n; ++i) {
      int idx = static_cast<int>(indices[i]);
      const float* wr = w.data() + idx * dim;
      float* orow = out.data() + i * dim;
      for (int d = 0; d < dim; ++d) orow[d] = wr[d];
    }
    return create_tensor(std::move(out));
  }

  // embedding_backward: scatter-add the upstream grad back into a (vocab x dim) dW:
  // dW[indices[i]] += dy[i], ACCUMULATING when an index repeats (sum-on-reuse, like
  // weight tying). Indices are not differentiable -- only W gets a gradient.
  inline int embedding_backward(int dy_id, std::vector<float> indices, int vocab, int dim) {
    const std::vector<float>& dy = tensors.at(dy_id);
    int n = static_cast<int>(indices.size());
    std::vector<float> dw(vocab * dim, 0.0f);

    for (int i = 0; i < n; ++i) {
      int idx = static_cast<int>(indices[i]);
      const float* dyr = dy.data() + i * dim;
      float* dwr = dw.data() + idx * dim;
      for (int d = 0; d < dim; ++d) dwr[d] += dyr[d];
    }
    return create_tensor(std::move(dw));
  }

} // namespace jabla
