# cpp/ — C++ for jank interop

jank's interop is **C++, not Java**, so JVM numeric libs (Neanderthal, Deep
Diamond, jblas) don't apply. Native numeric work reaches BLAS/cuBLAS/cuDNN
through the `cpp/` namespace. Headers and any vendored C++ sources live here,
alongside the jank that calls them.

## BLAS matmul (roadmap step 2 — done, CPU)
The native matmul kernel lives in `include/jabla.hpp` (`jabla::matmul`, a
`cblas_sgemm` over the registry buffers), called from `jabla.tensor` as
`(cpp/jabla.matmul ...)`. It's validated against the pure-jank `matmul-reference`
oracle (in `test/jabla/test_util.jank`) and C++ doctest cases in
`cpp/test/tensor_test.cpp`. The bring-up spike (`blas.jank`, element-at-a-time
global buffers) is retired now that the registry-native kernel works.

Remaining for the GPU port:
1. The first **host-vs-device timing split** — wall-clock around the call vs a
   CUDA-event timer inside it.
2. Move from CPU **`cblas_sgemm`** to **cuBLAS `cublasSgemm`** for the device
   split. cuBLAS handles/pointers/streams exercise the pointer/struct/handle
   territory where jank's interop edge cases surface.

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

## Layout
```
cpp/
  README.md        (this file)
  include/         jabla.hpp (tensor backend: registry + ops), doctest.h
  test/            tensor_test.cpp (doctest; run via `make cpp-test`)
```
Link flags reach the jank compile/JIT via `JANK_CPP_FLAGS` in the Makefile
(`-I cpp/include -L <blas-libdir> -lopenblas`), injected before the subcommand on
`run`/`test`/`repl`/`compile`. `make cpp-check` / `make cpp-test` build the
headers/tests locally with clang first (see `bin/cpp-toolchain`).
