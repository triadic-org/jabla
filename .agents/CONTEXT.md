# Context: jabla

## Overview
A from-scratch ML library in **jank** (Clojure on LLVM/C++). Name = jank + nabla (∇).
A learning project: port Karpathy's micrograd → nanoGPT ladder to jank to
understand models bottom-up. Build order + validators in `docs/roadmap.md`; jank
tooling notes/gotchas in `docs/jank-notes.md`; autograd design refs in
`docs/autograd-references.md`. Remote: `git@github.com:triadic-org/jabla.git`.

**Working mode:** the user writes the ML/jank code themselves to learn. The
assistant does scaffolding, boilerplate, tests, naming/comment passes, reviews,
and research — NOT the ML implementation unless explicitly asked.

## Current state (June 2026) — Steps 1-2 done, Step 3 in progress
- **Step 1 (scalar reverse-mode autograd): COMPLETE.** `src/jabla/autograd.jank`,
  tape design, green. Tests in `test/jabla/autograd_test.jank`: unit (per-op exact)
  + compositions + numerical `grad-check` + a micrograd oracle anchor.
- **Step 2 (native matmul via cpp/): COMPLETE (CPU).** `cpp/include/jabla.hpp`
  (`namespace jabla`: a registry of `std::vector<float>` buffers + `matmul` =
  `cblas_sgemm`), called from `jabla.tensor`. Validated vs `matmul-reference` (now
  in `test/jabla/test_util.jank`) + doctest in `cpp/test/tensor_test.cpp`. The
  bring-up spike (`blas.jank`, global element-at-a-time buffers) is **retired**.
- **Step 3 (tensors on native BLAS): IN PROGRESS.** `src/jabla/tensor.jank`:
  tensor = `{:shape :dtype :id}` over the C++ registry (bulk data stays in C++,
  marshaled only at the edges). `->tensor` / `reshape` (core.matrix-style) /
  `->vectors` and `matmul` forward are done + tested (square + rectangular). `add`
  is scaffolded (`jabla::add` C++ stub + RED tests). Remaining: `add` forward,
  then the **tensor tape** + `backward!` / `grad`.

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

## Next steps
- [ ] **First, re-verify on the devbox:** `make test`. The BLAS-spike retirement
      rewired the runner (dropped the `blas` suite) and moved `matmul-reference`
      into `test-util`; local lint/cpp-check are green, but the jank suite hasn't
      run since that change. Expect `autograd` + `tensor` green (`add-forward` is
      still a vacuous `(is true)` until `add` lands).
- [ ] Implement `jabla::add` (elementwise) in `jabla.hpp` + the jank `add` wrapper;
      turn the add tests green (`make cpp-test` + uncomment `add-forward`).
- [ ] Design + build the **tensor tape**: node = {output id, input ids, vjp};
      generalize the scalar tape to tensor payloads. Then `backward!` (matmul's vjp
      = two matmuls, routing back through `cpp/jabla.matmul`) + `grad`, validated
      with a tensor finite-difference grad-check.
- [ ] (later) Step 4: full GPT op set -> one attention block vs PyTorch -> nanoGPT.

> Local-only project notes (full brief, research angle) live in `private/`
> (gitignored, not published).
