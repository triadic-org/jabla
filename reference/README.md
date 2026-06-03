# reference/ — reference implementations

Karpathy's reference implementations, vendored to validate jabla numerically
(matching grads, loss curves, and intermediate activations).

These are **snapshots** (their `.git` was stripped so they don't become nested
repos). Provenance — re-clone upstream if you need history:

| Dir | Upstream | Pinned commit | Role |
|-----|----------|---------------|------|
| `micrograd/` | github.com/karpathy/micrograd | `c911406e5ace8742e5841a7e0df113ecb5d54685` | Scalar autograd (~150 lines). Step 1. |
| `makemore/`  | github.com/karpathy/makemore  | `988aa59e4d8fefa526d06f3b453ad116258398d4` | Tensor-level modeling. |
| `nanoGPT/`   | github.com/karpathy/nanoGPT   | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | GPT training (PyTorch — uses torch autograd, NOT micrograd). |
| `llm.c/`     | github.com/karpathy/llm.c     | `f1e2ace651495b74ae22d45d1723443fd00ecd3a` | GPT-2 in raw C/CUDA. **Manual forward+backward per layer.** Steps 2-5. |
| `llama2.c/`  | github.com/karpathy/llama2.c  | `350e04fe35433e6d2941dce5a1f53308f87058eb` | Single-file C inference of a Llama-2 transformer (forward only). |

Not vendored (aspirational, much later):
- **nanochat** (github.com/karpathy/nanochat) — full ChatGPT-style pipeline
  (tokenizer / pretrain / SFT / RL / inference). Scope map for post-pretraining;
  the Rust BPE tokenizer doesn't port to jank. Revisit past step 6.

## Usage
- **micrograd**: `micrograd/micrograd/engine.py` is the `Value` autograd;
  `micrograd/test/test_engine.py` checks grads against PyTorch — jabla's
  `jabla.autograd` should reproduce those numbers (see
  `../test/jabla/autograd_test.jank`).
- **llm.c**: `train_gpt2.c` (~1,180 lines, pure CPU fp32) is the key reference
  for the tensor era — one `gpt2_forward()`/`gpt2_backward()`, with an explicit
  `*_forward`/`*_backward` pair per layer (`gelu_*`, `layernorm_*`, `attention_*`,
  `matmul_*`). This is the manual-backprop math `jabla.tensor` must reproduce
  once you outgrow micrograd's auto-backward. `dev/cuda` develops each kernel in
  isolation — the pattern for the `cpp/` sgemm probe and beyond.
- **llama2.c**: `run.c` (~970 lines) is the cleanest minimal transformer
  *forward* pass in plain C (RMSNorm, RoPE, SwiGLU, KV cache) — reference for the
  `cpp/` forward path and a second architecture (Llama vs nanoGPT's GPT-2).
- **nanoGPT**: the loss-per-step curve on TinyShakespeare is the end-to-end
  numerical check. Log a reference curve into `../experiments/`, then match it.

## Re-cloning
```
git clone --depth 1 https://github.com/karpathy/micrograd.git
# rm -rf micrograd/.git to vendor it like these
```

Licensing: MIT (Karpathy). Keep the LICENSE files intact.
