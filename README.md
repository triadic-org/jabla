# jabla

An ML library in **[jank](https://jank-lang.org/)** — the native Clojure dialect
on LLVM with seamless C++ interop. The name is *jank + nabla (∇)*.

jank runs Clojure natively on LLVM and can call C++/CUDA directly through the
`cpp/` namespace, so orchestration and native numeric compute live in one
language and one REPL.

> **Status:** early — scaffold in place, functionality not yet implemented.

## Layout
```
src/jabla/      autograd.jank · tensor.jank · trace.jank · core.jank
test/jabla/     tests (clojure.test, which loads under jank) + test-util helpers
cpp/            C++ for BLAS/cuBLAS interop
reference/      reference implementations (vendored)
docs/           roadmap and tooling notes
experiments/    metrics & logs (gitignored contents)
data/           datasets, not committed (gitignored)
deps.edn        Clojure CLI — used to compute jank's --module-path
Makefile        wraps the jank CLI (see `make help`)
```

## Getting started
1. **Install jank** — see the [jank repo](https://github.com/jank-lang/jank);
   packaging is in flux, so building from source is likely. `make doctor`
   reports what's on your PATH.
2. **Run:** `make run` (executes `jabla.core/-main`).
3. **REPL:** `make repl`, then connect over nREPL (Conjure/CIDER).
4. **Tests:** `make test`.

With no external deps, `MODULE_PATH=src:test` works without a JDK/Clojure CLI;
you only need those once you add deps (`make module-path`).
