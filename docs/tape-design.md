# Tensor tape design — tensions, prior art, and what gates GPT

Notes for Step 3 (`jabla.tensor`, lift autograd to tensors) with the Step 4 GPT
goal in view. Companion to `docs/autograd-references.md` (which covers the
fast-tape *layout* — flat `vals[]`/`grads[]` + index tape — and the LLVM/C++ AD
engines); this doc is about the *design tensions* of making that tape tensor-aware
and whether any choice blocks the GPT endgame.

The scalar tape (`jabla.autograd`, Step 1) already settled the easy parts:
define-by-run, op-keyword nodes (data, not closures), reverse-scan backward,
flat parallel arrays. Tensors mostly generalize `:data` scalar -> tensor payload
and each op's local partial -> a vjp. The hard parts are below.

## The fault line

Almost every tension is the same collision: **a functional / immutable front end
(jank tensor maps `{:shape :dtype :id}`, a tape value) sitting on a mutable,
manually-managed native buffer registry (the C++ `tensors` vector, grows via
`push_back`; no per-tensor free, only a global `clearTensors`).** The tape is
exactly where those two worldviews meet —
it is ordered, stateful, owns lifetimes, accumulates in place — yet it wraps the
purest abstraction in the system. Get *ownership and identity* right and the rest
follows.

## The seven tensions

1. **Identity** — how does a tensor find its tape node? (a) node id carried on the
   tensor map, threaded through every op; or (b) tape keyed by buffer `:id`. (a)
   keeps the tensor honest but viral; (b) keeps the tensor minimal but couples
   buffer identity to node identity.
2. **Buffer lifetime vs graph lifetime** — backward needs forward activations
   alive (matmul's vjp needs A and B). Today nothing frees per-tensor (only a
   global `clearTensors`). Does the tape *own* intermediate buffers (and free them
   after backward) or merely *borrow* them?
3. **Define-by-run vs define-then-run** — record-as-side-effect (flexible, no
   graph-level optimization) vs a static graph (fusion/reorder/checkpoint, rigid).
   Step 1 chose by-run.
4. **vjp as closure vs data** — a closure capturing A/B is ergonomic but opaque and
   *implicitly pins buffers*; op-tag + rule table is inspectable, serializable, and
   matches the scalar tape. (The flat-tape layout in `autograd-references.md`
   already pushes us to data.)
5. **Gradient accumulation** — a tensor used twice gets two upstream grads that
   must *sum*. In-place accumulate (`saxpy`, fast, reintroduces mutation) vs
   functional sum (clean, allocates). Plus dense-zeros vs "missing == zero".
6. **C++/jank boundary** — kernel in C++, tape in jank is the right seam, but the
   vjp needs transposes + accumulation that are fast in BLAS yet natural in jank;
   every grad that crosses the boundary is marshalling + an ownership question.
7. **Higher-order / reentrancy** — a single global tape atom means backward can't
   be recorded, so no grad-of-grad and `backward!` is non-reentrant. Fine to
   accept, but it's a *decision*: a passed-value tape keeps the door open.

## How the prior art resolves them

| Tension | PyTorch | TF2 `GradientTape` | Deep Diamond (Clojure) |
|---|---|---|---|
| #1 Identity | `grad_fn` ptr on the tensor (implicit DAG, no flat tape) | explicit scoped tape *watches* tensors | none — the layer owns its state |
| #2 Lifetime | save *minimal* (`save_for_backward`), free graph after `.backward()`, hooks to offload activations | tape released per `.gradient()` (unless `persistent`) | **pre-allocate all buffers once, mutate in place** |
| #3 by-run/static | define-by-run | by-run (v2) / static `tf.Graph` (v1, enabled fusion/XLA) | **static layer graph** |
| #4 vjp form | codegen Node classes from `derivatives.yaml` (data) | `@RegisterGradient` op registry (data) | layer's own backward (data) |
| #5 accumulate | `AccumulateGrad`, sum in-place into `.grad` | tape aggregates | in-place into layer grad buffers |
| #6 boundary | all C++ engine, Python thin | all C++ engine | oneDNN/cuDNN kernels, Clojure orchestrates |
| #7 higher-order | `create_graph=True` re-tapes backward | nested tapes | not a goal |

Three points on the functional<->mutable spectrum:
- **PyTorch** — maximalist dynamic tape; node-on-tensor; aggressive minimal saving
  + freeing is the thing that makes it tractable at scale.
- **TF2 `GradientTape`** — the **tape as an explicit passed value**, scoped, nests
  for higher-order. This is the cleanest fit for a functional front end and is
  essentially the design recommended below.
- **Deep Diamond** (Dragan Djuric, the Clojure-over-native-buffers one) deliberately
  **rejects a general tape**: it's define-then-run specialized to NN layers on
  oneDNN/cuDNN, pre-allocating every forward/backward buffer once and mutating in
  place. Neanderthal underneath has *no* autodiff and a hard discipline of explicit
  scoped release (`with-release`/`release!`) — because a GC language must not leak
  off-heap buffers. That discipline is the direct lesson for jabla's #2/#6.
  (`clj-autograd` is the experiment that *did* port a PyTorch-style tape onto
  Neanderthal — proof it's possible in a Lisp, but it stayed an experiment.)

## What actually gates GPT (Step 4)

**Most tape decisions do not block GPT. Two do; one interacts; the real blockers
aren't tape decisions at all.**

### Tape decisions that gate GPT
- **#2 buffer lifetime — the hard wall.** "Never free" OOMs on the first
  non-toy model: a GPT forward holds every activation for backward and attention is
  **O(seq^2) memory per layer**. Real training needs **activation checkpointing**
  (recompute in backward instead of storing) — which is a *tape* capability the
  design must allow. Decide the ownership story now (**tape borrows; buffers freed
  after the consuming backward**) even if freeing is implemented later. Don't bake
  in "pin everything forever."
- **#5 accumulation — correctness, not just speed.** GPT **ties weights** (input
  embedding reused as output projection): one parameter, two uses, gradients must
  *sum*. An overwrite-instead-of-add accumulate path silently produces wrong grads.
  Get "used twice -> sum" right and tested before trusting any transformer training.

### Interacts (decide with GPT in mind)
- **#4 closure vs data — pick data.** A closure that captures A/B implicitly pins
  those buffers, which directly worsens #2 and forecloses checkpointing. Data /
  op-tag (already the scalar tape's choice, and what the flat-tape layout wants)
  keeps the engine free to decide save-vs-recompute.

### Does NOT gate GPT
- **#1 identity** — either model works at scale (PyTorch does node-on-tensor).
- **#3 by-run/static** — define-by-run trains every GPT in the world; don't build a
  static graph just for GPT (a transformer's static shape *would* let a graph
  compiler fuse, but that's an optimization, not a blocker).
- **#7 higher-order** — vanilla GPT training never needs grad-of-grad.

### The real blockers — none are tape decisions
1. **GPU.** CPU OpenBLAS (`cblas_sgemm`) can *inference* a small GPT, not
   meaningfully *train* one. Keep the kernel layer **behind a dispatchable
   interface** so a cuBLAS/Metal backend slots in; never hardcode `cblas_*` inside
   vjp rules (that couples autodiff to CPU). (Ties to the Step 2 cuBLAS port and
   the host-vs-device timing split already in the roadmap.)
2. **dtype.** `std::vector<float>` / `:dtype :f32` everywhere. GPT wants bf16/fp16.
   The tensor map already carries `:dtype` (extensible); the C++ registry is the
   refactor. Storage decision, independent of the tape.
3. **2D-only matmul.** `matmul(a,b,m,n,k)` is strictly 2D; attention is **batched**
   matmul over (batch, heads, seq, dim), and its vjp differs from the 2D rule. Ship
   2D now, just don't let the 2D vjp assumption ossify.
4. **The reshape-in-pure-jank pattern is a scaling trap.** `reshape` in pure jank is
   fine; softmax / layernorm / GELU / attention written the same way would be orders
   of magnitude too slow. **Rule going in: every hot op is a native kernel; jank only
   orchestrates and records the tape.**

## Recommended direction

- **Tape as a passed value** (not a global atom), **vjp-as-data keyed by op**
  (consistent with the scalar tape and the flat-array layout), **buffers borrowed,
  freed after the backward that consumes them** (interface allows checkpointing even
  before it's implemented). This is essentially TF2's `GradientTape` + gradient
  registry, heeding Neanderthal's explicit-ownership discipline.
- **Kernels behind a dispatchable interface; never pin buffers in closures.** Those
  two properties are what keep the GPU / dtype / batched-kernel work (the real
  blockers) unblocked later.

The strategic fork to stay aware of: **general vjp tape** (flexible, allocate-heavy,
PyTorch-shaped) **vs static layer graph** (fast, rigid, Deep-Diamond-shaped). The
learning goal favors the general tape; the performance endgame may pull toward
pre-allocated layer state. The recommended design is GPT-compatible and
checkpointing-ready without committing to either extreme yet.

## Sources
- PyTorch autograd: <https://github.com/pytorch/pytorch/blob/main/docs/source/autograd.rst>,
  saved-tensors hooks: <https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html>
- Deep Diamond: <https://dragan.rocks/articles/19/Fast-tensors-Clojure-sneak-peek>,
  <https://github.com/uncomplicate/deep-diamond>
- clj-autograd (PyTorch-style tape on Neanderthal): <https://github.com/whilo/clj-autograd>
