# Roadmap

Module build order, each with a validator so it's always clear whether jank or
the implementation is at fault.

| # | Step | Lives in | Validator | Status |
|---|------|----------|-----------|--------|
| 1 | Scalar reverse-mode autograd | `src/jabla/autograd.jank` | grads match `reference/micrograd/test/test_engine.py` (vs PyTorch) | **Done** |
| 2 | Single BLAS/cuBLAS `sgemm` via `cpp/` | `cpp/` + `src/jabla/blas.jank` | matmul matches a reference; first host-vs-device timing split | **Done (CPU)** [1] |
| 3 | Lift autograd to tensors on native BLAS | `src/jabla/tensor.jank` | one attention block matches PyTorch element-wise | **Scaffolded** [2] |
| 4 | Full GPT; train TinyShakespeare → TinyStories | new ns (e.g. `jabla.gpt`) | loss-per-step curve in `experiments/` tracks nanoGPT's | Todo |
| 5 | Instrument MFU / tokens-sec | timing utils | wall-clock vs CUDA-event; MFU vs a PyTorch yardstick | Todo |
| 6 | Additional methods (e.g. SAEs) | new ns | reproduce the reference metric | Todo |

_Status as of 2026-06-04._
- **[1]** CPU `cblas_sgemm` (OpenBLAS) matches `blas/matmul-reference` via `make test`;
  the host-vs-device timing split is deferred to the cuBLAS/GPU port. Step 1 is
  validated by `test/jabla/autograd_test.jank` (per-op unit grads + finite-difference
  grad-check + a micrograd `test_sanity_check` oracle).
- **[2]** `tensor.jank` has the representation, autograd-via-tape plan, API stubs,
  and a milestone order; the op forwards and tensor `backward!` are unimplemented.

`src/jabla/trace.jank` (eligibility traces) is an independent track, not part of
the GPT order above.

Ground truth lives in `reference/`: micrograd (step 1, grads vs PyTorch), llm.c's
`train_gpt2.c` (steps 2-5, manual per-layer forward+backward), llama2.c's `run.c`
(clean forward pass). When iterating on the autograd/tensor engine itself, see
`docs/autograd-references.md` (fast-tape design + LLVM/C++ AD frameworks).

## Measurement
Always log, for each shape: tokens/sec and step time; the **host (wall-clock) vs
device (CUDA-event) split** (separates jank-runtime cost from implementation
cost); and a PyTorch reference number for the same shape. Write these to
`experiments/` (gitignored contents) so curves accumulate.

## Dataset ladder
TinyShakespeare (bring-up) → TinyStories/SimpleStories (does it learn?) →
FineWeb-Edu 10B (small pretrain) → WikiText-103 (comparable perplexity).
Datasets are downloaded into `data/` (gitignored) — never committed.
