# jank — tooling notes (June 2026)

Ground-truth checks on jank's tooling. jank is alpha and moving fast —
**re-verify against the current docs before relying on this.**

## Status
- jank entered **alpha in January 2026**. Q2 2026 focus is performance +
  continuous benchmarking. The object-model transition is still in progress.

## Build tooling
jank uses the **Clojure CLI** as its build tool.
- Source files in `src/`, tests in `test/`, standard Clojure project layout.
- `deps.edn` works; git / maven / local deps resolve "like a normal Clojure
  project" (but most JVM libs won't *load* — interop is C++, not Java).
- The Clojure CLI computes `--module-path`: `clojure -Spath`.

Example commands from the jank docs:
```
# REPL with test paths on the module path
jank --module-path $(clojure -A:test -Spath) repl

# run a main
jank --module-path $(clojure -Spath) run-main jank-cli-test.core 1
```
Here, `deps.edn` handles path/dep resolution and the Makefile wraps the jank CLI
calls. With no external deps the module path is just `src:test`, so a JDK /
Clojure CLI isn't needed to start — the Makefile defaults `MODULE_PATH=src:test`.

## Install — in flux (don't hardcode)
As of March 2026, packaging is unresolved: LLVM 22 was expected to drop the
vendored Clang/LLVM requirement and enable package-manager distribution, but
that **didn't land** (API-compat + perf regressions); the Arch binary package
was reported broken. No confirmed Homebrew tap in the sources checked.
→ Expect **build-from-source**; check the jank repo's current install docs.

## C++ interop
- All native symbols live under the reserved `cpp/` namespace.
- C/C++ **includes can be part of the `ns` macro**; follows ClojureScript's
  `:refer-global` for bringing native symbols in.
- `cpp/cast` (≈ `static_cast`), `cpp/unsafe-cast` (≈ C-style cast).
- Manual memory via `cpp/new` / `cpp/delete` (bdwgc GC underneath).
- A **type-encoding DSL** supports arbitrary C++ types: templates, NTTPs, refs,
  pointers, const/volatile, pointers-to-members, pointers-to-functions — the
  territory cuBLAS/cuDNN APIs live in. Edge cases there still surface bugs.
- Typed (polymorphic) C++ exception catching works.

## REPL / editor
- **nREPL server works** — tested with NeoVim/Conjure and Emacs/CIDER.
- Native AOT gives sub-100ms startup; both static and dynamic executables.

## Known gaps
- No user-facing **test framework** → `test/jabla/test_harness.jank` (hand-rolled).
- No **clj-kondo** for `cpp/` forms → lint blind spots around interop.

## Sources
- jank clojure-cli docs: https://github.com/jank-lang/jank/blob/main/clojure-cli/README.md
- jank repo / site: https://github.com/jank-lang/jank · https://jank-lang.org/
- "jank is off to a great start in 2026": https://jank-lang.org/blog/2026-03-06-great-start/
- Interop series: https://jank-lang.org/blog/2025-05-02-starting-seamless-interop/ ·
  https://jank-lang.org/blog/2025-06-06-next-phase-of-interop/ ·
  https://jank-lang.org/blog/2025-07-11-jank-is-cpp/
