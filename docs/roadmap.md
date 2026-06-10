# Roadmap

Module build order, each with a validator so it's always clear whether jank or
the implementation is at fault.

| # | Step | Lives in | Validator | Status |
|---|------|----------|-----------|--------|
| 1 | Scalar reverse-mode autograd | `src/jabla/autograd.jank` | grads match `reference/micrograd/test/test_engine.py` (vs PyTorch) | **Done** |
| 2 | Single BLAS/cuBLAS `sgemm` via `cpp/` | `cpp/include/jabla.hpp` | matmul matches a reference; first host-vs-device timing split | **Done (CPU)** [1] |
| 3 | Eager explicit-graph autograd engine (node-on-tensor) | `src/jabla/tensor.jank` | tensor finite-diff grad-check; weight-tying grads sum; one attention block matches PyTorch element-wise | **Stage 1 done** [2] |
| 4 | Full GPT; train TinyShakespeare → TinyStories | new ns (e.g. `jabla.gpt`) | loss-per-step curve in `experiments/` tracks nanoGPT's | Todo |
| 5 | Instrument MFU / tokens-sec | timing utils | wall-clock vs CUDA-event; MFU vs a PyTorch yardstick | Todo |
| 6 | Additional methods (e.g. SAEs) | new ns | reproduce the reference metric | Todo |

_Status as of 2026-06-09._
- **[1]** CPU `cblas_sgemm` (OpenBLAS) matches `matmul-reference` (in `test-util`)
  via `make test`. The bring-up spike (`blas.jank`) is retired now that the kernel
  lives natively in `cpp/include/jabla.hpp` (`jabla::matmul`), called from
  `jabla.tensor`; the host-vs-device timing split is deferred to the cuBLAS/GPU
  port. Step 1 is validated by `test/jabla/autograd_test.jank` (per-op unit grads +
  finite-difference grad-check + a micrograd `test_sanity_check` oracle).
- **[2]** Step 3 was reframed from a micrograd-to-matrices port to a frontier-shaped
  **eager explicit-graph engine** (node-on-tensor DAG, vjp-as-data, lazy/fused/multi-
  backend later); see `docs/engine-design.md` for the decision + staged roadmap.
  **Stage 1 (eager autograd) is done + green:** the `matmul`/`add` forwards (recording
  `:op`/`:inputs`), the `:add`/`:matmul` vjp rules, and `backward!` (`seed-grads` ->
  `topo-order` -> vjp dispatch -> sum-on-reuse accumulation) + `get-grad`, validated
  by finite-diff grad-checks incl. a linear-layer composition + the weight-tying gate.
  Remaining for Step 3: the rest of the op set (mul, relu/gelu, softmax, layernorm,
  embedding, cross-entropy) -> one attention block vs PyTorch (the row's validator);
  engine stages (arena/free, lazy, fusion, backend) per `engine-design.md`.

`src/jabla/trace.jank` (eligibility traces) is an independent track, not part of
the GPT order above.

Ground truth lives in `reference/`: micrograd (step 1, grads vs PyTorch), llm.c's
`train_gpt2.c` (steps 2-5, manual per-layer forward+backward), llama2.c's `run.c`
(clean forward pass). When iterating on the autograd/tensor engine itself, see
`docs/autograd-references.md` (fast-tape layout + LLVM/C++ AD frameworks) and
`docs/engine-design.md` (the tensor engine architecture: chosen node-on-tensor
direction, design tensions, staged eager -> lazy -> fused roadmap, and the
PyTorch/TF2/tinygrad/ggml prior art).

## Measurement
Always log, for each shape: tokens/sec and step time; the **host (wall-clock) vs
device (CUDA-event) split** (separates jank-runtime cost from implementation
cost); and a PyTorch reference number for the same shape. Write these to
`experiments/` (gitignored contents) so curves accumulate.

## Dataset ladder
TinyShakespeare (bring-up) → TinyStories/SimpleStories (does it learn?) →
FineWeb-Edu 10B (small pretrain) → WikiText-103 (comparable perplexity).
Datasets are downloaded into `data/` (gitignored) — never committed.
