# Context: jabla

## Overview
A from-scratch ML library in **jank** (Clojure on LLVM/C++). Name = jank + nabla (∇).
A learning project: port Karpathy's micrograd → nanoGPT ladder to jank to
understand models bottom-up. Build order + validators in `docs/roadmap.md`; jank
tooling notes/gotchas in `docs/jank-notes.md`; autograd design refs in
`docs/autograd-references.md`; the tensor engine architecture (chosen direction,
design tensions, staged roadmap, prior art) in `docs/engine-design.md`.
Remote: `git@github.com:triadic-org/jabla.git`.

**Working mode:** the user writes the ML/jank code themselves to learn. The
assistant does scaffolding, boilerplate, tests, naming/comment passes, reviews,
and research — NOT the ML implementation unless explicitly asked.

## Direction (decided 2026-06-08) — Step 3 reframed
Step 3 is **not** a micrograd-to-matrices port (the scalar work already taught the
chain rule). It's building a **frontier-shaped autograd engine**, complexity added
progressively. After reasoning end-first (best architecture = deferred explicit
graph-as-IR over dispatchable kernels — JAX/tinygrad/ggml), we chose the **middle
path**: author eager now, but reify the graph as a value so deferred/fused/multi-
backend stays open. Representation = **node-on-tensor DAG** (the tensor map carries
`:op` + `:inputs`; the graph IS the immutable tensor DAG, like PyTorch grad_fn /
tinygrad _ctx) — no global tape, no threaded tape. vjp-as-data keyed by `:op`.
`backward!` returns grads (no global); `grad` is a lookup. GAN/actor-critic aren't
foreclosed — PyTorch does them via `detach` + separate optimizers (a future op),
not multiple tapes. Full decision + the staged eager→lazy→fused→backend roadmap:
`docs/engine-design.md`.

## Current state (June 2026) — Steps 1-2 done, Step 3 stage 1 (eager autograd) done
- **Step 1 (scalar reverse-mode autograd): COMPLETE.** `src/jabla/autograd.jank`,
  tape design, green. Tests in `test/jabla/autograd_test.jank`: unit (per-op exact)
  + compositions + numerical `grad-check` + a micrograd oracle anchor.
- **Step 2 (native matmul via cpp/): COMPLETE (CPU).** `cpp/include/jabla.hpp`
  (`namespace jabla`: a registry of `std::vector<float>` buffers + `matmul` =
  `cblas_sgemm`), called from `jabla.tensor`. Validated vs `matmul-reference` (now
  in `test/jabla/test_util.jank`) + doctest in `cpp/test/jabla_test.cpp`. The
  bring-up spike (`blas.jank`, global element-at-a-time buffers) is **retired**.
- **Step 3 (tensor autograd engine): STAGE 1 COMPLETE (eager).** `src/jabla/tensor.jank`:
  tensor = `{:shape :dtype :id :op :inputs}` over the C++ registry (bulk data in C++,
  marshaled only at the edges). All green:
  - **forwards** `->tensor`/`reshape`/`->vectors`, `matmul` + `add` -- each is a
    node-free `*-raw` kernel call + `(assoc … :op … :inputs …)`. matmul has CblasTrans
    flags (the vjp uses them; no materialized transpose). `add` is rank-agnostic.
  - **vjp-rules** registry: `:add` (grad straight through) + `:matmul` (`dA = dY·Bt`,
    `dB = At·dY` via transposed `matmul-raw`s).
  - **eager backward**: `seed-grads` (ones tensor at the root), `topo-order`
    (post-order DFS, dedup by id, reverse-topo), `backward!` (reduce over the order,
    dispatch vjp by `:op`, accumulate into inputs **summing on reuse** via `add-raw`),
    `get-grad` (lookup by id).
  - **tests** (`tensor_test.jank`): forwards, vjp rules in isolation, topo-order
    (diamond w/ shared non-leaf), backward smokes (add/chain/weight-tying, exact
    ones/twos), finite-diff grad-checks (add, matmul, weight-tying, a linear-layer
    `matmul`+`add` composition). C++ doctests (`jabla_test.cpp`) incl. transposed matmul.

## C++ interop (settled)
- All native code in `cpp/include/jabla.hpp` under `namespace jabla`. jank calls
  it **qualified** — `(cpp/jabla.matmul ...)` -> `jabla::matmul(...)`. The namespace
  is load-bearing: a `(defn matmul)` would shadow a global C++ `matmul` (jank emits
  unqualified calls), but a qualified call can't be shadowed. So both sides keep
  natural names. Full rule + the lazy-stdlib gotchas in `docs/jank-notes.md`.
- **C++ naming = snake_case (settled 2026-06-11):** `create_tensor`, `get_tensor`,
  `softmax_backward`, etc., to read continuously with the STL the kernels lean on.
  Collision-safety is the namespace (above), not the case, so it's purely stylistic;
  the jank side keeps kebab-case. (`_` survives jank munging unchanged.)
- C++ tested with doctest (`make cpp-test`); header syntax via `make cpp-check`;
  `make check` (= ASCII + lint + cpp-check) runs in the pre-commit hook.

## Naming convention (settled)
- **Scalar tape (`jabla.autograd`):** `node` = tape entry (`:leaf` | op-result);
  `value` = the `{:id i}` handle. append primitives bang (`push-node!`/`push-leaf!`);
  readers get- (`get-data`/`get-grad`/`get-node-data`); builders no affix
  (`add`/`mul`/`pow*`/…); lifecycle bang (`reset-tape!`/`backward!`).
- **Tensor engine (`jabla.tensor`):** the tensor map IS the node (node-on-tensor;
  no separate `value`). Role names for the *same* map: `t`/`t1`/`t2` generic, `node`
  in the vjp/walk, `root` at backward's entry, `grad`/`grad-out` for gradient tensors.
  Readers get- (`get-shape`/`get-dtype`/`get-grad`). Node-free kernels suffix `-raw`
  (`matmul-raw`/`add-raw`); public op = `*-raw` + `(assoc … :op … :inputs …)`. No
  global tape, so no `reset-tape!`; `backward!` returns the grads map.
- Comments: one style — `;; --- title ---` + plain `;;` prose (no boxes).

## Tooling
- jank 0.1-alpha installed via Ubuntu PPA. `make run|test|repl|doctor|health`.
- Python venv `.venv/` (gitignored): torch 2.12.0+cpu + numpy + pytest (runs the
  micrograd reference for ground truth).
- clj-kondo lint + pre-commit hook: `make lint`, `make hooks` (hook = ASCII guard
  + clj-kondo; per-clone, run `make hooks` after cloning).
- jank gotchas: ASCII-only source (lexer rejects non-ASCII even in comments);
  no `--version` (use `check-health`); no `(catch :default ...)`; don't shadow
  core names you call.

## Op set progress (the Step 3 op track)
Each op = forward (records `:op`/`:inputs` via a `*-raw` kernel) + a vjp rule + tests.
Split settled: assistant does the mechanical copies, user does the learning ones.
- **Done (assistant): `mul`, `gelu`.** Elementwise; C++ kernels (`mul`, `gelu` tanh
  approx + fused `geluBackward`) + jank forwards + `:mul`/`:gelu` vjp rules + tests.
- **Done (user): `relu`.** `relu` + `reluBackward` C++ kernels (kink `relu'(0):=0`) +
  forward/backward/no-alias doctests (`cpp/test/jabla_test.cpp`); jank scaffold ready.
- **Sequencing (re-decided 2026-06-10): order by NEAREST validator, not op difficulty.**
  Goal = reach "one attention block validated element-wise vs PyTorch" (Step 3 finish)
  fast -- the cheapest end-to-end test of the DAG + backward engine. Reframe: all
  remaining ops are on GPT's critical path (no wide-vs-deep tension *among them*); they
  split by *which* milestone they gate.
    1. `softmax` (coupled reduction -- attention core; also used by cross-entropy) <- NEXT
    2. `layernorm` (reuses softmax's row-reduction machinery; pre-norm)
    3. the engine work the block forces -- **this is where to "go wide"**: batched/N-D
       matmul (vjp differs from the 2-D rule), a causal mask, broadcast/scalar-scale
       (1/sqrt(d_k) + layernorm affine). softmax/layernorm themselves stay just-enough.
    --> assemble + validate the attention block (Step 3 done).
    4. `cross-entropy` (loss; needs softmax) + `embedding` (input layer) come AFTER --
       they gate the trainable GPT (Step 4), NOT the block, so deferred, not skipped.
  Scaffolds (commented, uncomment-as-you-go): jank targets in `tensor_test.jank` (all
  five ops); C++ doctests through `softmax`/`softmaxBackward` in `jabla_test.cpp`.
  Gotchas baked in: softmax/layernorm grad-check is VACUOUS under plain `sum` (rows sum
  to 1 / to beta) -> weight the output (`(mul (softmax x) C)`); embedding indices aren't
  differentiable (hand-oracle the scatter-add).
- **Future-proofing checked (2026-06-10): the GPT-first sequence forecloses no future
  architecture (GANs/RL/meta-learning).** Those stress autograd *topology*, a different
  axis from the forward-op/shape work on the GPT path. Deferred axes now captured in
  `engine-design.md` ("Deferred axes" table): `detach` (cheap future op), multiple-
  backward-from-different-roots (already supported -- stateless `backward!`), conv (just
  another registry op). The ONE real watch-item is **double-backward (`create_graph`)**
  for WGAN-GP/MAML: today's vjps route through node-free `*-raw`, so grads aren't
  differentiable. Not foreclosed (flip vjps to node-recording under a flag, PyTorch-
  style); guardrail: never let "a grad tensor has no `:op`/`:inputs`" become load-bearing,
  and keep `-raw`-vs-node-recording a swappable choice.

## Other tracks / open threads
- **Engine stages (perf/scale; see `engine-design.md`):** `arena/free` (registry never
  frees -> a training loop OOMs; needed before Step 4) -> lazy realize -> fusion ->
  backend dispatch (cuBLAS/Metal). Plus a `detach` op for GAN/actor-critic.
- **Decided this session:** keep `autograd.jank` (Step-1 scalar reference, not dead
  code); defer splitting `tensor.jank` / `jabla.hpp` (engine vs ops/kernels) until
  softmax+layernorm land and the boundary is clear; the `*-raw` -> public-op wrapper
  could become a `defop` macro later (function `as-node` is the 80% if wanted now).
- **Then** Step 4 (full GPT, train TinyShakespeare) and Step 5 (MFU/timing).

> Local-only project notes (full brief, research angle) live in `private/`
> (gitignored, not published).
