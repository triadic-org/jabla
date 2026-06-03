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

## Current state (June 2026) — Step 1 done, Step 2 scaffolded
- **Step 1 (scalar reverse-mode autograd): COMPLETE.** `src/jabla/autograd.jank`
  is a tape design, fully implemented and green. Tests layered in
  `test/jabla/autograd_test.jank`: unit (per-op exact) + compositions (neg/sub/div)
  + integration (numerical `grad-check` over arbitrary exprs, incl. the full
  micrograd test_more_ops) + one exact oracle anchor (sanity-check). **44 passed,
  1 pending** (`make test`).
- **Step 2 (sgemm via cpp/): SCAFFOLDED, not implemented.** `src/jabla/blas.jank`
  has a done pure-jank `matmul-reference` (ground truth) + a guided `sgemm` stub.
  `test/jabla/blas_test.jank`: reference test live, sgemm-vs-reference test pending.

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
- [ ] Implement `blas/sgemm` via `cpp/` (cblas_sgemm) — the C++ interop is the
      learning task. Two unknowns: how jank links OpenBLAS, and the jank↔C float*
      marshaling. Fallback: inline-C++ matmul first (no external lib), then swap
      in cblas_sgemm. Then flip the pending blas test green + add host/device timing.
- [ ] (later) Step 3: lift autograd to tensors on native BLAS (`tensor.jank`).

> Local-only project notes (full brief, research angle) live in `private/`
> (gitignored, not published).
