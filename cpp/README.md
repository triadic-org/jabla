# cpp/ — C++ for jank interop

jank's interop is **C++, not Java**, so JVM numeric libs (Neanderthal, Deep
Diamond, jblas) don't apply. Native numeric work reaches BLAS/cuBLAS/cuDNN
through the `cpp/` namespace. Headers and any vendored C++ sources live here,
alongside the jank that calls them.

## The `sgemm` probe (roadmap step 2)
The first interop milestone is a single **`sgemm`** (single-precision matmul)
through `cpp/`:
1. Correctness — result matches a reference matmul.
2. The first **host-vs-device timing split** — wall-clock around the call vs a
   CUDA-event timer inside it.

Start with **CPU BLAS** (OpenBLAS/Accelerate `cblas_sgemm`) to prove the interop
shape with no GPU, then move to **cuBLAS `cublasSgemm`** for the device split.
cuBLAS handles/pointers/streams exercise the pointer/struct/handle territory
where jank's interop edge cases surface.

## Interop primitives (verify against your jank version — see ../docs/jank-notes.md)
- `cpp/` namespace for native symbols; C/C++ **includes can go in the `ns` macro**.
- `cpp/new` / `cpp/delete` for manual memory (bdwgc GC underneath).
- `cpp/cast` (≈ `static_cast`), `cpp/unsafe-cast` (≈ C-style cast) — the latter
  for messy C APIs (handles, `void*`).
- A type-encoding DSL for arbitrary C++ types (templates, pointers, refs, …).

Interop syntax is young; reference the interop blog series:
https://jank-lang.org/blog/2025-05-02-starting-seamless-interop/ ·
https://jank-lang.org/blog/2025-06-06-next-phase-of-interop/ ·
https://jank-lang.org/blog/2025-07-11-jank-is-cpp/

## Layout (suggested)
```
cpp/
  README.md        (this file)
  include/         vendored headers, if any
  src/             vendored .cpp, if any
```
Link flags (`-lopenblas`, `-lcublas`, include dirs) need to reach the jank
compile/JIT — capture them in the Makefile once the flag-passing mechanism for
your jank build is known.
