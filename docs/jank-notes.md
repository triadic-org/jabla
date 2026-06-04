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
- **Include headers via `(:include "foo.h")` in the `ns` macro** (verified
  2026-06-04): the idiomatic form; multiple strings allowed, and
  `(:refer-global :only [...])` brings symbols in unqualified. The older
  `(cpp/raw "#include <foo.h>")` also works (use it for one-off snippets).
- `cpp/cast` (≈ `static_cast`), `cpp/unsafe-cast` (≈ C-style cast).
- Manual memory via `cpp/new` / `cpp/delete` (bdwgc GC underneath).
- A **type-encoding DSL** supports arbitrary C++ types: templates, NTTPs, refs,
  pointers, const/volatile, pointers-to-members, pointers-to-functions — the
  territory cuBLAS/cuDNN APIs live in. Edge cases there still surface bugs.
- Typed (polymorphic) C++ exception catching works.

### Interop gotchas (verified 2026-06-04, building the BLAS sgemm path)
- **Only primitives cross the boundary cleanly.** A jank double/int converts to
  C++ and back automatically. C++ objects/references do NOT: an expression
  returning e.g. `std::ostream&` gives `analyze/invalid-conversion: ... not
  convertible to a jank runtime object`. Run side-effecting statements via
  `(cpp/raw "...;")` so nothing crosses back.
- **Heavily-overloaded C++ functions don't resolve.** `(cpp/std.sqrt 16.0)` fails
  `analyze/invalid-cpp-function-call: ambiguous` even with `(cpp/cast cpp/double
  ...)` — a jank value matches several overloads and the resolver doesn't fully
  tie-break. Prefer single-signature functions (your own wrappers, or C APIs like
  `cblas_*`); reach for `cpp/cast` only when forced.
- **C++ UB hard-crashes the whole jank process** (the JIT runs in-process), so a
  `std::vector` out-of-bounds via `operator[]` is a silent segfault. Use `.at()`
  while bringing interop up so it throws a *catchable* C++ exception instead; swap
  back to `[]` once indexing is trusted.
- **Header-only helpers: mark functions/globals `inline`** (C++17) — standard
  header-only idiom; one shared instance, no ODR issues across the translation
  units jank's JIT creates.

## Building / linking native libs (verified 2026-06-04)
jank CLI flags go **before** the subcommand: `jank -I <dir> -L <dir> -lfoo run ...`.
- `-I` / `--include-dir` for headers (system headers in default dirs, e.g.
  `cblas.h` in `/usr/include/x86_64-linux-gnu`, are found without `-I`).
- `-L` / `--library-dir` and `-l<name>` to link a shared lib.
- **jank does NOT search the default linker/ldconfig paths.** A bare `-lopenblas`
  fails with `Failed to load dynamic library`; pass `-L <dir> -lopenblas` (or a
  full path: `-l /lib/x86_64-linux-gnu/libopenblas.so`). Find the dir with
  `ldconfig -p | grep libopenblas.so`.
- Wired in the Makefile as `JANK_CPP_FLAGS` (`-I cpp/include -L $(CBLAS_LIBDIR)
  -lopenblas`) on `run`/`test`/`repl`/`compile`, with a `check-blas` preflight
  that points at `make deps` if the lib is missing.

## REPL / editor
- **nREPL server works** — tested with NeoVim/Conjure and Emacs/CIDER.
- Native AOT gives sub-100ms startup; both static and dynamic executables.

## Testing
- **`clojure.test` works under jank** — bundled in the runtime, loads with no
  extra deps (`deftest`/`is`/`are`/`testing`/`use-fixtures`/`run-tests`).
  `run-tests` takes ns symbols, so per-file running is built in
  (`make test SUITE=autograd`). Pass CLI args to `-main` after a `--` separator.
  - Caveat: failure headers print `(nil) (:)` — jank doesn't capture the deftest
    var name / line (metadata gap), so put the human label in a `(testing "...")`
    block (it prints on failure). Float tolerance: a local `approx?` helper in
    `test/jabla/test_util.jank` (clojure.test has none).

## Known gaps
- No jank-specific **linter/formatter**. We lint with **clj-kondo** (native binary,
  no JVM) treating `.jank` as `clj` — `make lint` mirrors the sources to `.clj`
  for project-wide analysis. Config in `.clj-kondo/config.edn`. The one blind
  spots are jank's `cpp/` interop forms and its extended `ns` clauses
  (`:include` / `:refer-global`) — handled in `.clj-kondo/config.edn` by excluding
  `cpp` from `:unresolved-namespace` and turning off `:unknown-ns-option`.
  `make hooks` installs a pre-commit hook that runs `make check` (ASCII + lint +
  `cpp-check`).

## Sources
- jank clojure-cli docs: https://github.com/jank-lang/jank/blob/main/clojure-cli/README.md
- jank repo / site: https://github.com/jank-lang/jank · https://jank-lang.org/
- "jank is off to a great start in 2026": https://jank-lang.org/blog/2026-03-06-great-start/
- Interop series: https://jank-lang.org/blog/2025-05-02-starting-seamless-interop/ ·
  https://jank-lang.org/blog/2025-06-06-next-phase-of-interop/ ·
  https://jank-lang.org/blog/2025-07-11-jank-is-cpp/
