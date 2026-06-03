# Autograd references

Notes to draw on when iterating on `jabla.autograd` / `jabla.tensor` past the
first naive port. Two threads: how to make a hand-rolled autograd *fast*, and
whether to lean on an existing LLVM/C++ AD engine instead.

## 1. Making a hand-rolled autograd fast (Rogozhnikov, 2023)

Source: <https://arogozhnikov.github.io/2023/12/28/fastest-autograd.html>
(author of einops). Scenario: many *changing* graphs, 10k-100k nodes each, run
~10k times forward+backward — latency-bound, not throughput-bound. His ladder:

| Impl | Time |
|---|---|
| PyTorch (est.) | ~1130 s |
| JAX + JIT (est.) | ~470 s |
| Python, object-per-node tape | 312 s |
| Python, **flat arrays + indices** | 94 s |
| Python + Numba on that layout | 41 s |
| Rust / C inner loop | ~1 s |

The lessons that transfer to jabla:
- **Tape with flat parallel arrays beats an object graph.** Store all values in
  one `vals[]` and gradients in a parallel `grads[]`; ops reference *indices*,
  not node objects. This is the single biggest structural win (object-graph →
  index-tape was ~3x in pure Python before any compilation).
- **Backward = a simple reverse iteration over the recorded tape.** No graph
  walk, no per-node closures — just replay ops in reverse, accumulating grads.
- **Delegate the hot loop to compiled code.** The 40x step (Numba) and 100x+
  step (Rust/C) come from running the tape replay in a compiled inner loop.
- **Pitfalls:** JIT/compile overhead can dominate for changing graphs (JAX spent
  >99% compiling; `torch.compile` crashed on large graphs). Don't pay
  whole-graph compilation cost per iteration when the graph keeps changing.

### Why this matters specifically for jabla
- It **validates strategy (c)** sketched in `src/jabla/autograd.jank` — a single
  tape/graph in one atom (flat data + indices), not a tree of `deftype`/`atom`
  nodes. Start with the object version to learn the chain rule, then move to the
  tape; that move is where the perf is.
- jank's host is young/slow, so the object-allocation overhead the post measures
  hits us *harder*. The flat-tape layout is also exactly what hands cleanly to
  `cpp/` later: a `vals[]`/`grads[]` float buffer + an op-tape is a native inner
  loop waiting to happen — the same "delegate the hot loop" move, via jank's C++
  interop instead of Numba/Rust.
- Tape replay in reverse is trivially the structure `jabla.tensor` wants too
  (each op records its local vjp into the tape).

## 2. Existing LLVM / C++ autograd frameworks

Yes — and two of them sit unusually close to jank's substrate and ecosystem.

### The two that matter for jank
- **Enzyme** — <https://github.com/EnzymeAD/Enzyme> (enzyme.mit.edu). AD that
  operates on **LLVM/MLIR IR**, differentiating *already-optimized* IR, which is
  why it matches/beats source-level AD tools. Language-agnostic (anything that
  lowers to LLVM: C/C++/Rust/Fortran/Julia/Swift) and works on GPU/parallel code.
  **jank compiles to LLVM IR**, so Enzyme is conceptually the AD engine that
  could differentiate jank's *own* substrate — or, more practically, the C++/CUDA
  kernels you call through `cpp/`. This is the most jank-aligned AD tool that
  exists.
- **Clad** — <https://github.com/vgvassilev/clad>. A **Clang plugin** doing
  source-transformation AD on the C++ AST (forward + reverse), emitting C++
  derivative code. Notably it's from Vassil Vassilev's compiler-research group —
  the same lineage as Cling / CppInterOp, the C++-interpreter tech jank's interop
  is built on. So Clad lives in *jank's exact toolchain ecosystem*; if you're
  already JIT-compiling C++ via jank, Clad-style differentiation of those
  functions is a natural fit.

### The operator-overloading / expression-template family (classic C++ AD)
General-purpose, library-level (no compiler plugin), usable from `cpp/`:
- **autodiff** (autodiff.github.io) — modern C++17, forward + reverse, dual
  numbers / expression templates.
- **CppAD**, **ADOL-C** — mature operator-overloading AD (optimization world).
- **Stan Math** — reverse-mode via expression templates (powers Stan).
- **Adept** — fast reverse-mode, expression templates (Hogan).
- **Ceres `Jet`** — forward-mode dual numbers in Google's Ceres solver.
- **libtorch / ATen** — PyTorch's own C++ autograd engine (heavy, but it *is* the
  industrial reference for a tape-based reverse-mode engine in C++).

### Strategic note (honest framing)
For the **learning goal**, hand-roll it — the whole point is to feel the chain
rule and manual per-layer backward (that's what `reference/llm.c` spells out).
But there's a real, jank-flavored research thread here: jank is on LLVM, and
Enzyme differentiates LLVM IR. "Could jank + Enzyme give native AD over code
written in a Lisp, on the GPU?" is a genuine question that the usual
Python/PyTorch world can't ask. Park it as a later experiment, not a shortcut
around the learning — but it's the kind of thing that makes the jank substrate
choice pay off beyond pedagogy.
