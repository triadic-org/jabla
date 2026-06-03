# Roadmap

Module build order, each with a validator so it's always clear whether jank or
the implementation is at fault.

| # | Step | Lives in | Validator |
|---|------|----------|-----------|
| 1 | Scalar reverse-mode autograd | `src/jabla/autograd.jank` | grads match `reference/micrograd/test/test_engine.py` (vs PyTorch) |
| 2 | Single BLAS/cuBLAS `sgemm` via `cpp/` | `cpp/` + a probe ns | matmul matches a reference; first host-vs-device timing split |
| 3 | Lift autograd to tensors on native BLAS | `src/jabla/tensor.jank` | one attention block matches PyTorch element-wise |
| 4 | Full GPT; train TinyShakespeare → TinyStories | new ns (e.g. `jabla.gpt`) | loss-per-step curve in `experiments/` tracks nanoGPT's |
| 5 | Instrument MFU / tokens-sec | timing utils | wall-clock vs CUDA-event; MFU vs a PyTorch yardstick |
| 6 | Additional methods (e.g. SAEs) | new ns | reproduce the reference metric |

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
