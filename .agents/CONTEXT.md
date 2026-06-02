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

## Environment state (June 2026)
- Installed: git, make, network. **NOT installed:** jank, clojure CLI, java.
- jank install is in flux (LLVM 22 packaging unresolved) → expect build-from-source.

## Next Steps
- [ ] Install jank (`make doctor` to check); confirm `make run`/`make repl` work.
- [ ] Verify the exact jank CLI subcommands in the Makefile against the installed
      version (`run-main`, `compile-module`, process-exit idiom in test_runner).
- [ ] Implement `jabla.autograd` (validate against `reference/micrograd`).
- [ ] Stand up the `sgemm` probe in `cpp/` for the host/device timing split.

> Local-only project notes live in `private/` (gitignored, not published).
