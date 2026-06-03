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

## Install
**There is an official Ubuntu PPA** (supports 24.04 / 24.10 / 25.04) — far
faster than building from source (the source path needs **LLVM 22**, which isn't
in Ubuntu's apt, so it compiles LLVM from scratch: 1–2 hours).

Verified working on Ubuntu 24.04 (June 2026):
```bash
sudo apt install -y curl gnupg
curl -s "https://ppa.jank-lang.org/KEY.gpg" | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/jank.gpg >/dev/null
sudo curl -s -o /etc/apt/sources.list.d/jank.list "https://ppa.jank-lang.org/jank.list"
sudo apt update && sudo apt install -y jank
```
After install, `jank check-health` (≡ `make health`) confirms it can JIT-compile
C++ and AOT-compile binaries. Source-build instructions:
https://github.com/jank-lang/jank/blob/main/compiler+runtime/doc/build.md

## Gotchas found locally (jank 0.1-alpha)
- **The CLI has no `--version`** — use `jank check-health` for diagnostics, or
  `jank --help` for the version banner + subcommands (`run`, `run-main`, `repl`,
  `cpp-repl`, `compile`, `compile-module`, `check-health`).
- **The lexer rejects non-ASCII bytes — even inside comments and strings**
  (`lex/invalid-unicode: Unfinished character`). Keep `.jank` source ASCII-only:
  no em-dashes, no `∇`/`λ`/`±`, etc. (Markdown docs are unaffected.)
- **Don't shadow `clojure.core` names you also call.** Defining `(defn reset! …)`
  that internally calls core `reset!` resolves to your own fn (arity mismatch →
  `JIT session error: Symbols not found`). Name helpers distinctly (we use
  `clear!`).

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
