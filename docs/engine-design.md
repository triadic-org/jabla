# Tensor autograd engine — direction, tensions, prior art

Design notes for the `jabla.tensor` autograd engine, with the Step 4 GPT goal in
view. Companion to `docs/autograd-references.md` (fast-tape *layout* — flat
`vals[]`/`grads[]` + index tape — and the LLVM/C++ AD engines); this doc is about
the *architecture* of the tensor engine and which choices block the endgame.

The scalar tape (`jabla.autograd`, Step 1) already settled the easy parts:
define-by-run, op-keyword nodes (data, not closures), reverse-scan backward,
flat parallel arrays. The scalar work also *was* the chain-rule learning -- so the
tensor engine is deliberately **not** a micrograd-to-matrices port. It aims at a
frontier-shaped engine (an explicit graph that can later go lazy + fused +
multi-backend), introducing that complexity progressively. The decision and its
reasoning are below; the seven tensions and prior art that informed it follow.

## Decision (2026-06-08): the middle path

After reasoning from "what's the best autograd architecture in the abstract"
(deferred, explicit graph-as-IR over dispatchable kernels -- JAX / tinygrad / ggml)
down to jabla's goals, we chose the **middle path**:

> **Author eager, but reify the graph as a value** -- eager execution now (simple,
> debuggable), with the graph a first-class structure rather than ambient/global
> state, so the door to deferred execution + fusion + a dispatchable backend stays
> open at near-zero cost today.

Concretely:
- **Representation (A): node-on-tensor DAG** (PyTorch / tinygrad shape). The tensor
  map carries its provenance -- `:op` (an op keyword) and `:inputs` (the input
  tensors) -- and the *graph is the immutable tensor DAG itself*. No global atom, no
  threaded tape; the graph is already a value, and the same DAG becomes the lazy
  graph later. (Rejected (B): a standalone IR object -- ggml `cgraph` / JAX
  `jaxpr`-style -- as more machinery than stage 1 needs; lift toward it only if a
  later fusion pass wants a cleaner target.)
- **vjp-as-data**, keyed by `:op` in a rule registry (consistent with the scalar
  tape; the only choice that keeps the engine free to save-vs-recompute later).
- **No global tape**: `backward!` walks from a root tensor and returns the grads;
  `grad` is a lookup. `reset-tape!` is gone.
- **GAN / actor-critic are not foreclosed**: PyTorch trains them on an ambient
  graph via `detach` (graph-cut) + separate optimizers, *not* multiple tapes. The
  unlock jabla needs for them is a future **`detach` op**, noted in the roadmap.

Why eager-ambient-global-atom was *rejected* despite being simplest: an ambient
global graph is not a value you can hand to a scheduler/compiler, so it forecloses
the deferred-execution evolution (it's exactly why `torch.compile` had to bolt on
a tracing layer to *extract* a graph PyTorch never reified). Node-on-tensor keeps
that path open.

## Staged roadmap (end-first)

End state: explicit graph-as-IR, lazily realized, fused kernels, arena memory,
dispatchable multi-backend. The increments back from there, each on that path
(not throwaway):

| Stage | What | Early simplification (and its later home) |
|---|---|---|
| 1 | **[DONE 2026-06-09] Eager backward over the node-on-tensor DAG.** Ops execute eagerly *and* record `:op`/`:inputs`; `backward!` walks reverse-topo from a root, dispatches vjp by `:op`, accumulates per buffer (sum on reuse); rules for `add` (passthrough) + `matmul` (two matmuls via CblasTrans). Validated via finite-diff grad-checks (add, matmul, linear-layer) + the weight-tying used-twice-sums test. | -- (this is the core engine learning) |
| 2 | **Lifetime / arena.** A C++ `freeTensor` / region so backward frees consumed activations (the Neanderthal / ggml discipline; #2). | skip -- `clearTensors` between steps -- until memory bites (-> here) |
| 3 | **Lazy realize.** Ops stop executing eagerly; they record into the DAG, and `realize` / `->vectors` triggers compute. The seam that unlocks fusion -- the whole reason stage 1 reifies the graph. | eager (-> here) |
| 4 | **Fusion + scheduling.** Fuse elementwise chains / softmax / attention (this is where the O(seq^2) attention-memory wall below actually gets solved -- flash-attention is a fusion). | -- |
| 5 | **Backend dispatch.** Kernel interface; CPU now, CUDA / Metal later. | single CPU backend (-> here) |

Orthogonal storage/kernel refactors slot in when needed: bf16/fp16 dtype, batched
/ N-D matmul (attention is batched over (batch, heads, seq, dim) and its vjp differs
from the 2-D rule), and a `detach` op for GAN / actor-critic.

### Deferred axes (known doors, not walls)

These are *not* on the GPT critical path, so they're deliberately not built yet --
but each is something a future architecture (GANs, RL, meta-learning) will want, and
the point of recording them is that the node-on-tensor + vjp-as-data substrate keeps
every one a **door** (add an op / flip a flag) rather than a **wall** (re-architect).
The guardrails below cost nothing today; their only job is to stop an assumption from
quietly ossifying against them.

| Axis | Who needs it | Status / unlock | Guardrail to keep it open |
|---|---|---|---|
| **`detach` / stop-gradient** | GANs (detach fakes for the D step), actor-critic (detach the value target) | A future op: result is a fresh `:leaf` that drops its `:inputs` edge, so `backward!` stops there. Orthogonal to every other op. | Nothing special -- the DAG already cuts cleanly at a `:leaf`. |
| **Double-backward (`create_graph`)** | WGAN-GP / R1 gradient penalty, MAML / meta-learning -- anything that puts a gradient *inside* the loss | The vjp rules currently route through node-free `*-raw` kernels, so **grads are not themselves differentiable**. Unlock: let the vjps optionally route through the *node-recording* ops (so grad tensors carry their own `:op`/`:inputs`), under a `create-graph` flag -- PyTorch's exact move. The representation already supports it. | (1) Never let "a grad tensor has no `:op`/`:inputs`" become load-bearing anywhere (`backward!`, `get-grad`, future arena/lazy code). (2) Treat "vjp uses `-raw` vs node-recording op" as a swappable choice, not an invariant. This is tension #7 made concrete. |
| **Multiple backward from different roots** | GANs (separate D-loss / G-loss), any multi-objective setup | **Already supported** -- `backward!` is stateless and returns a grads map; the DAG is a persistent value, so calling backward from two roots Just Works. | -- (a property of the no-global-tape decision; don't add global state that breaks it.) |
| **New layer kernels (conv, etc.)** | DCGAN and most non-transformer nets | Each is just another op = forward (`-raw` + record) + a vjp rule in the registry. The engine never changes. | -- (this *is* the extensibility the vjp-as-data registry buys.) |

The throughline: GANs/RL stress the autograd **topology** (how many backward passes,
which edges are cut), which is a different axis from the forward-op + shape work on
the GPT path -- so the GPT-first op sequence forecloses none of them, *provided* the
double-backward guardrails above hold.

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

| Tension | PyTorch | TF2 `GradientTape` | Deep Diamond (Clojure) | tinygrad | ggml |
|---|---|---|---|---|---|
| #1 Identity | `grad_fn` ptr on the tensor (implicit DAG, no flat tape) | explicit scoped tape *watches* tensors | none — the layer owns its state | `_ctx` (Function) on the tensor — node-on-tensor | explicit `ggml_cgraph` you build then compute |
| #2 Lifetime | save *minimal* (`save_for_backward`), free graph after `.backward()`, hooks to offload activations | tape released per `.gradient()` (unless `persistent`) | **pre-allocate all buffers once, mutate in place** | lazy buffers realized on demand; scheduler plans memory | **arena**: alloc in a `ggml_context`, bulk-free the context |
| #3 by-run/static | define-by-run | by-run (v2) / static `tf.Graph` (v1, enabled fusion/XLA) | **static layer graph** | **lazy** — record, then `realize()` (fuses) | define-then-run static graph |
| #4 vjp form | codegen Node classes from `derivatives.yaml` (data) | `@RegisterGradient` op registry (data) | layer's own backward (data) | `Function.backward` (data) | `ggml_build_backward_expand` (data) |
| #5 accumulate | `AccumulateGrad`, sum in-place into `.grad` | tape aggregates | in-place into layer grad buffers | sums into `.grad` | inference-first; backward secondary |
| #6 boundary | all C++ engine, Python thin | all C++ engine | oneDNN/cuDNN kernels, Clojure orchestrates | Python thin, **codegen'd kernels per backend** | all C, backends as structs |
| #7 higher-order | `create_graph=True` re-tapes backward | nested tapes | not a goal | not a focus | not a goal |

tinygrad and ggml are the two references that pushed the chosen direction: tinygrad
shows the autograd itself is **tiny and node-on-tensor** (the real engineering is
the lazy graph + per-backend codegen — keep autograd small, invest in the execution
layer); ggml shows the **explicit-graph + arena-memory** end (graph as a value,
region allocation, kernels are the product). jabla's node-on-tensor + eager-now /
lazy-later is tinygrad's substrate; the arena lifetime story (stage 2) is ggml's.

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

> Superseded by the **Decision (2026-06-08)** at the top, which this section
> originally argued toward. Kept for the reasoning trail. One correction: it framed
> the graph-form choice as *passed-value tape* (TF2) vs global atom -- but the
> chosen **node-on-tensor DAG** is a third option that is neither. Node-on-tensor
> is *already* a value (no global, no threading) and is the eager-natural,
> lazy-ready substrate (tinygrad), which the passed-value framing undersold. The
> rest below still holds.

- **Graph as a value** (chosen: node-on-tensor DAG), **vjp-as-data keyed by op**
  (consistent with the scalar tape and the flat-array layout), **buffers borrowed,
  freed after the backward that consumes them** (interface allows checkpointing even
  before it's implemented), heeding Neanderthal's explicit-ownership discipline.
- **Kernels behind a dispatchable interface; never pin buffers in closures.** Those
  two properties are what keep the GPU / dtype / batched-kernel work (the real
  blockers) unblocked later.

The strategic fork to stay aware of: **general vjp graph** (flexible, allocate-heavy,
PyTorch/tinygrad-shaped) **vs static layer graph** (fast, rigid, Deep-Diamond-shaped).
The learning goal favors the general graph; the performance endgame may pull toward
pre-allocated layer state. The chosen design is GPT-compatible and
checkpointing-ready without committing to either extreme yet.

## Sources
- PyTorch autograd: <https://github.com/pytorch/pytorch/blob/main/docs/source/autograd.rst>,
  saved-tensors hooks: <https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html>
- Deep Diamond: <https://dragan.rocks/articles/19/Fast-tensors-Clojure-sneak-peek>,
  <https://github.com/uncomplicate/deep-diamond>
- clj-autograd (PyTorch-style tape on Neanderthal): <https://github.com/whilo/clj-autograd>
- tinygrad (node-on-tensor autograd + lazy graph + per-backend codegen):
  <https://github.com/tinygrad/tinygrad>
- ggml (explicit `cgraph` + arena memory, all-C, inference-first):
  <https://github.com/ggml-org/llama.cpp> (and the ggml library it vendors)
