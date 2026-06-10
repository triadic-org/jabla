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
| **core.typed** | static type checker | **No** — bound to the JVM analyzer (`tools.analyzer.jvm`); also largely dormant upstream | **Skip.** Not probed (JVM-bound by construction). |
| **clojure.spec.alpha** | runtime spec/validation + gen | **No (probed)** — ships as `.clj`; jank's loader only reads `.cljc`/`.jank`, so it can't even be found | **Can't use.** See empirical results below. |
| **malli** | runtime schema-as-data + humanized errors + gen | **No (probed)** — `.cljc` files are found, but jank's lexer aborts inside `malli/impl/util.cljc` | **Can't use.** See below. |
| **plumatic/schema** | runtime schema-as-data + coercion | **No (probed)** — `.cljc` found, but `deftype` (in `schema.utils`) isn't ported in jank-alpha | **Can't use.** See below. |
| **hand-rolled (`jabla.schema`)** | runtime predicates + asserts | **Yes** — bedrock core only, zero deps | **Adopt.** The only option that runs today; covers all three goals. |
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

## Empirical results (probed 2026-06-10, on the devbox, jank 0.1-alpha)

All three external libs were fetched (`git clone`) and `require`d under jank. **None
load.** Three *independent* root causes, each on its own sufficient to block:

1. **Loader reads only `.cljc` / `.jank`, not `.clj`.** Verified with a 3-file probe
   (`foo.bar.clj` / `.cljc` / `.jank`): the `.clj` require fails `module-not-found`,
   the other two load. So `clojure.spec.alpha` (ships as `clojure/spec/alpha.clj`) and
   any lib whose required namespaces are `.clj` (e.g. `schema.macros`) are unreachable.
2. **`.cljc` reads under a `:jank` reader-conditional branch** — not `:clj`. Verified:
   `#?(:jank :jank :clj :clj :default :other)` evaluates to `:jank`. Existing libs guard
   their host code behind `:clj` / `:cljs`, *neither of which fires under jank* — so even
   a findable `.cljc` silently loses its interop / macro bodies. (Fixing this needs the
   **upstream library** to add `:jank` branches, not just a maturing jank.)
3. **jank-alpha hasn't ported the foundations these libs are built on** — `deftype`,
   and the lexer/reader isn't yet complete enough for malli's source.

Per-lib outcome:

| Lib | Failure (exact) |
|---|---|
| **clojure.spec.alpha** | `module-not-found` — it's a `.clj` (cause #1). Dead on arrival. |
| **plumatic/schema** | `analyze/unresolved-symbol: Unable to resolve symbol 'deftype'` in `schema/utils.cljc` (cause #3). Schema is built on deftype/protocols throughout. |
| **malli** | `lex/invalid-number: Unexpected end of integer` in `malli/impl/util.cljc` (cause #3) — jank's alpha lexer can't even read malli's source. |

### Reproduce

```bash
# Loader-extension + reader-conditional probes are inline above; lib probes:
cd /tmp && git clone --depth 1 https://github.com/clojure/spec.alpha.git spec
git clone --depth 1 https://github.com/plumatic/schema.git schema
git clone --depth 1 https://github.com/metosin/malli.git malli
git clone --depth 1 https://github.com/borkdude/dynaload.git dynaload   # malli's one runtime dep

cd <repo>
jank --module-path "experiments/typing:/tmp/spec/src/main/clojure"        run-main jabla.spec-probe
jank --module-path "experiments/typing:/tmp/schema/src/cljc:/tmp/schema/src/clj" run-main jabla.schema-probe
jank --module-path "experiments/typing:/tmp/malli/src:/tmp/dynaload/src"  run-main jabla.malli-probe
```
(Probe namespaces live in `experiments/typing/jabla/*_probe.jank` — gitignored; each is
a one-line `require` whose abort IS the data point. The lexer/`deftype` failures are
genuine jank-alpha bugs/gaps worth filing upstream.)

## Recommendation

1. **Adopt `jabla.schema`** — it is not a fallback, it is the *only* runtime-schema
   option that loads on jank today. Add `:pre` / `check` at the API edges — `->tensor`
   (valid nested input), `matmul`/`add`/`mul` (`tensor?` + shape compatibility),
   `backward!` (valid root). Keep it opt-in: on at the edges and while bringing an op up;
   it can be flagged/compiled off the hot path later.
2. **Re-probe when jank matures** (track the lexer + `deftype` gaps). But note cause #2:
   even a fixed jank won't run malli/schema until *they* ship `:jank` reader-conditional
   branches — so the realistic external-lib path is "jank gains a `:clj`-compatible mode"
   or "a jank-native schema lib appears," not "malli just works." Don't block on it.
3. **Skip core.typed.** And keep leaning on **clj-kondo** as the static net.

Net: external schema libs are **off the table on jank-alpha** — confirmed, not assumed.
And even if one loaded, the dependent **shape** checks — the bugs that actually hurt —
are a hand-written predicate in every world; a lib would only prettify the *structural*
half and add generators. So `jabla.schema` isn't a compromise floor; it's both the
high-value part and the only part that runs.

## Sources / pointers
- malli: <https://github.com/metosin/malli>
- clojure.spec: <https://clojure.org/guides/spec>
- core.typed (why it's JVM-bound / dormant): <https://github.com/clojure/core.typed>
