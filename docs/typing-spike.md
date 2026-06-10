# Typing / schema-validation spike (branch `spike/typing`, 2026-06-10)

Goal (from the ask, in priority order): **schema validation**; helping a developer
**reason about what each step expects**; and **potentially catching bugs early**.
Static typing per se is *not* the goal.

## The jank constraint (why this isn't a normal Clojure spike)

- **jank runs only on the devbox** (Ubuntu PPA); the dev box is where `make test`
  runs. Locally there's no jank and **no Clojure CLI**, so malli/spec can't even be
  *resolved* here, let alone loaded. Loadability is inherently a **devbox** check.
- `deps.edn` already warns: *"most JVM Clojure libs won't load under jank (host is
  LLVM/C++, interop is C++ not Java)."* jank is alpha and ports `clojure.core`
  piecemeal. So "does malli/spec load at all?" is the gating question, not ergonomics.

That flips the spike: build the thing that is **guaranteed** to work, and prepare a
fast devbox probe for the maybes.

## Options

| Tool | Kind | Works on jank? | Verdict |
|---|---|---|---|
| **core.typed** | static type checker | **No** — bound to the JVM analyzer (`tools.analyzer.jvm`); also largely dormant upstream | **Skip.** Don't probe. |
| **clojure.spec.alpha** | runtime spec/validation + gen | **Unknown** — ships with Clojure, but jank may not have ported `spec.alpha` | Probe (`experiments/typing/spec_probe.jank`). Fallback if malli won't load. |
| **malli** | runtime schema-as-data + humanized errors + gen | **Unknown** — mostly pure Clojure, but uses protocols/regex/etc. jank may lack | Probe (`experiments/typing/malli_probe.jank`). The nicest *if* it loads. |
| **hand-rolled (`jabla.schema`)** | runtime predicates + asserts | **Yes** — bedrock core only, zero deps | **Adopt now.** The safe baseline; covers all three goals. |
| **clj-kondo** | static analysis | **Yes** — already in the pipeline | Lean on it for what it catches (see below). |

## The "catch bugs at compile time" reality

True static *type* checking on jank isn't available (core.typed is the only real
option and it doesn't load). But you already have the practical equivalents:

- **clj-kondo is your compile-time net** — it's static, JVM-native (no jank needed),
  and already in `make check` / the pre-commit hook. It catches **arity, unresolved
  vars, unused bindings across namespaces** — and it has already paid off (it caught
  the `topo-order` arity mismatch). It will *not* catch structural/shape errors.
- **Structural + shape errors are runtime** — caught by `jabla.schema` (or malli) at
  the op boundaries, plus the finite-diff grad-checks for numerical correctness.
- **Shape errors are dependent-typed** ("matmul's `k` dims must match", "add needs
  equal shapes"). No mainstream Clojure type/schema lib expresses these ergonomically
  — they're a custom predicate either way (`jabla.schema/matmul-compatible?`). This
  is the highest-value check (a bad shape currently segfaults the C++ kernel via
  `.at()` or silently mismatches), and it's the same code in any approach.

## What's in this branch

- **`src/jabla/schema.jank`** — the hand-rolled layer. `tensor?` (structural),
  `explain-tensor` / `check` (problem-listing + throwing), `same-shape?` /
  `matmul-compatible?` (the dependent-shape checks). Pure jank, lints clean.
  `test/jabla/schema_test.jank` covers + documents it.
- **`experiments/typing/{malli,spec}_probe.jank`** — devbox loadability probes
  (off the build/lint path). Each builds a `Tensor` schema and validates a good +
  bad value, so a successful run also shows the ergonomics.

### Devbox probe steps

```bash
# malli:
#   add to deps.edn :deps ->  metosin/malli {:mvn/version "0.16.4"}
make module-path
jank --module-path "$(clojure -A:test -Spath):experiments/typing" run-main jabla.malli-probe

# spec (no dep needed if jank bundles it):
jank --module-path src:test:experiments/typing run-main jabla.spec-probe
```
A require that aborts = that lib doesn't load under jank (a useful data point; maybe
a jank issue to file). A clean run prints the validation + humanized/explain output.

## Recommendation

1. **Adopt `jabla.schema` now.** Add `:pre` / `check` at the API edges — `->tensor`
   (valid nested input), `matmul`/`add`/`mul` (`tensor?` + shape compatibility),
   `backward!` (valid root). Keep it opt-in: on at the edges and while bringing an op
   up; it can be flagged/compiled off the hot path later. This satisfies all three
   goals today with zero risk.
2. **Run the two probes on the devbox** (~10 min). If **malli loads**, it's a strict
   upgrade for *errors + generative testing* — migrate `jabla.schema`'s structural
   part to a malli `Tensor` schema, but keep the shape-compat predicates as code
   (malli won't express them). If only **spec loads**, same idea, lower ergonomics.
   If **neither loads**, stay on `jabla.schema` — you lose nothing essential.
3. **Skip core.typed.** And keep leaning on **clj-kondo** as the static net.

Net: the dependent **shape** checks — the bugs that actually hurt — are a hand-written
predicate in every world. malli/spec would only prettify the *structural* half and add
generators. So the floor (`jabla.schema`) is already most of the value; the probes just
tell us whether to put a nicer face on it.

## Sources / pointers
- malli: <https://github.com/metosin/malli>
- clojure.spec: <https://clojure.org/guides/spec>
- core.typed (why it's JVM-bound / dormant): <https://github.com/clojure/core.typed>
