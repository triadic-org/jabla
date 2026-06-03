# Context: jabla

## Overview
An ML library in **jank** (Clojure on LLVM/C++). Name = jank + nabla (∇). The
module build order and validators are in `docs/roadmap.md`; verified jank
tooling notes in `docs/jank-notes.md`.

## Current Work
**Scaffold only — not yet implemented.** Repo structure, Makefile/deps.edn build
wiring, docs, hand-rolled test harness, and vendored reference implementations
are in place. Source files (`src/jabla/*.jank`) are stubs that sketch API +
design notes.

## Key Decisions
- **Build tooling:** Clojure CLI `deps.edn` (computes `--module-path`) wrapped by
  a Makefile (jank CLI invocations). `MODULE_PATH=src:test` works with no JDK
  until real deps are added. Details + sources in `docs/jank-notes.md`.
- **No jank test framework** → hand-rolled assertions in
  `test/jabla/test_harness.jank` + a `test_runner`.
- **C++ interop syntax intentionally not scaffolded** — it's young/easy to get
  wrong; `cpp/README.md` points to the interop docs.

## Environment state (June 2026) — toolchain is up
- **jank 0.1-alpha installed** via the official Ubuntu PPA (`make doctor` → found).
  `make health` (`jank check-health`) is green: JIT C++ + AOT both work.
- **`make run` and `make test` both pass** through real jank.
- **Python validation venv** at `.venv/` (gitignored): torch 2.12.0+cpu + numpy.
  `reference/micrograd` PyTorch ground-truth tests pass (`pytest test/`).
- Not installed (not needed yet): clojure CLI + JDK — only for `make module-path`
  once external deps are added.
- jank gotchas captured in `docs/jank-notes.md`: ASCII-only source, no `--version`
  (use `check-health`), don't shadow core names you call.

## Next Steps
- [ ] Implement `jabla.autograd` (validate against `reference/micrograd`;
      ground-truth tests already run green in the venv).
- [ ] Stand up the `sgemm` probe in `cpp/` for the host/device timing split
      (jank JIT-compiles C++, so interop is viable).
- [ ] Wire real assertions into `test/jabla/autograd_test.jank` as ops land.

> Local-only project notes live in `private/` (gitignored, not published).
