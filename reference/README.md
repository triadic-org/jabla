# reference/ — reference implementations

Karpathy's reference implementations, vendored to validate jabla numerically
(matching grads, loss curves, and intermediate activations).

These are **snapshots** (their `.git` was stripped so they don't become nested
repos). Provenance — re-clone upstream if you need history:

| Dir | Upstream | Pinned commit | Role |
|-----|----------|---------------|------|
| `micrograd/` | github.com/karpathy/micrograd | `c911406e5ace8742e5841a7e0df113ecb5d54685` | Scalar autograd (~150 lines). |
| `makemore/`  | github.com/karpathy/makemore  | `988aa59e4d8fefa526d06f3b453ad116258398d4` | Tensor-level modeling. |
| `nanoGPT/`   | github.com/karpathy/nanoGPT   | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | GPT training. |

Optional, later:
- **llm.c** (github.com/karpathy/llm.c) — the C/CUDA version. `git clone
  --depth 1` into `reference/llm.c/` when needed.

## Usage
- **micrograd**: `micrograd/micrograd/engine.py` is the `Value` autograd;
  `micrograd/test/test_engine.py` checks grads against PyTorch — jabla's
  `jabla.autograd` should reproduce those numbers (see
  `../test/jabla/autograd_test.jank`).
- **nanoGPT**: the loss-per-step curve on TinyShakespeare is the end-to-end
  numerical check. Log a reference curve into `../experiments/`, then match it.

## Re-cloning
```
git clone --depth 1 https://github.com/karpathy/micrograd.git
# rm -rf micrograd/.git to vendor it like these
```

Licensing: MIT (Karpathy). Keep the LICENSE files intact.
