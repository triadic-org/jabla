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

## Current state (June 2026) — Steps 1-2 done, Step 3 (engine, stage 1) in progress
- **Step 1 (scalar reverse-mode autograd): COMPLETE.** `src/jabla/autograd.jank`,
  tape design, green. Tests in `test/jabla/autograd_test.jank`: unit (per-op exact)
  + compositions + numerical `grad-check` + a micrograd oracle anchor.
- **Step 2 (native matmul via cpp/): COMPLETE (CPU).** `cpp/include/jabla.hpp`
  (`namespace jabla`: a registry of `std::vector<float>` buffers + `matmul` =
  `cblas_sgemm`), called from `jabla.tensor`. Validated vs `matmul-reference` (now
  in `test/jabla/test_util.jank`) + doctest in `cpp/test/jabla_test.cpp`. The
  bring-up spike (`blas.jank`, global element-at-a-time buffers) is **retired**.
- **Step 3 (tensors on native BLAS): IN PROGRESS.** `src/jabla/tensor.jank`:
  tensor = `{:shape :dtype :id}` over the C++ registry (bulk data stays in C++,
  marshaled only at the edges). `->tensor` / `reshape` (core.matrix-style) /
  `->vectors` and the **`matmul` + `add` forwards** are done + tested. matmul
  validated vs `matmul-reference` (square + rectangular); `add` is the elementwise
  `jabla::add` kernel (no BLAS) + jank wrapper, green in both the C++ doctest and
  the jank suite. `add` is rank-agnostic (passes the input shape straight through,
  unlike matmul's 2-D `[m n]`); the buffer accessors are a by-value-sink
  `createTensor` (move-in) + reference reads from the registry. The forwards now
  also **record their graph node** (`:op` + `:inputs`; leaves are `:op :leaf`).
  Remaining (stage 1): `vjp-rules` (`:add` passthrough, `:matmul` two-matmuls +
  transposes), `backward!` (reverse-topo walk + vjp dispatch + sum-on-reuse), and
  `grad` -- all scaffolded as contract stubs in `tensor.jank` (bodies are yours).

## C++ interop (settled)
- All native code in `cpp/include/jabla.hpp` under `namespace jabla`. jank calls
  it **qualified** — `(cpp/jabla.matmul ...)` -> `jabla::matmul(...)`. The namespace
  is load-bearing: a `(defn matmul)` would shadow a global C++ `matmul` (jank emits
  unqualified calls), but a qualified call can't be shadowed. So both sides keep
  natural names. Full rule + the lazy-stdlib gotchas in `docs/jank-notes.md`.
- C++ tested with doctest (`make cpp-test`); header syntax via `make cpp-check`;
  `make check` (= ASCII + lint + cpp-check) runs in the pre-commit hook.

## Naming convention (settled)
- **node** = tape entry (`:leaf` | op-result); **value** = the `{:id i}` handle
  user code holds. ("value" is the accurate scalar term; "tensor" is reserved for
  Step 3 arrays.)
- append primitives bang: `push-node!`, `push-leaf!`. readers get-: `get-data`,
  `get-grad`, `get-node-data`. builders no affix: `add`/`mul`/`pow*`/`relu`/`neg`/
  `sub`/`div`, `->value` (coercion). lifecycle bang: `reset-tape!`, `backward!`.
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

## Next steps (stage 1 -- eager backward over the node-on-tensor DAG)
Scaffolding is in place (contract stubs in `tensor.jank`; `tu/tensor-grad-check` +
three RED-but-vacuous deftests in `tensor_test.jank` -- `add-backward`,
`matmul-backward`, `weight-tying-sums`). The engine logic is the user's to write:
- [ ] Populate `vjp-rules`: `:add` (pass grad-out through to both inputs), then
      `:matmul` (`dA = dY . Bt`, `dB = At . dY` -- both back through
      `cpp/jabla.matmul`). Open sub-decision: transposes from sgemm's `CblasTrans`
      flag (free) vs a separate transpose op.
- [ ] Implement `backward!` (seed ones at root; reverse-topo walk via `:inputs`;
      dispatch vjp by `:op`; accumulate per buffer id, **summing on reuse**) and
      `grad` (look up by `(:id t)` in the returned grads).
- [ ] Turn the three deftests green: uncomment bodies, wire `add-backward` first
      (vjp is trivial -- best for wiring the walk), then `matmul-backward`, then
      `weight-tying-sums` (the sum-on-reuse correctness gate). Adjust the helper's
      two API calls if your `backward!`/`grad` signatures differ.
- [ ] Re-verify on the devbox: `make test` + `make cpp-test`.
- [ ] (later stages, see `docs/engine-design.md`) arena/free -> lazy realize ->
      fusion -> backend dispatch; plus a `detach` op for GAN/actor-critic, and the
      rest of the GPT op set toward one attention block vs PyTorch.

> Local-only project notes (full brief, research angle) live in `private/`
> (gitignored, not published).
