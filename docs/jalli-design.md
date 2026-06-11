# jalli -- a schema-as-data validation layer for jabla

`jabla.jalli` is a tiny, malli-flavored runtime validation layer: **schemas are data,
errors are data**, and the engine is a multimethod. It exists because the established
Clojure schema libraries (malli, clojure.spec, plumatic/schema) **do not load on
jank-alpha** -- see `docs/typing-spike.md` for the empirical proof. jalli keeps the two
ideas worth keeping from malli and drops the one jank can't run.

## Why not just use malli? (the constraint that shapes everything)

Probed on the devbox (full evidence in `docs/typing-spike.md`):

- jank's module loader reads `.cljc` / `.jank`, **not `.clj`** -- and `.cljc` is read
  under a `:jank` reader-conditional branch, so any code a lib guards behind `:clj`
  never fires.
- jank-alpha has **no `defprotocol` / `defrecord` / `deftype` / `reify`** -- but it
  **does** have `defmulti` / `defmethod`, `case`, and the full map/vector/keyword core.

malli and plumatic/schema are built on protocols + records, so their *architecture* is
impossible here, not just inconvenient. What IS available -- `defmulti` over data -- is
exactly enough to rebuild malli's *interface*. So jalli is malli's good ideas on jank's
available primitives.

## The model

A **schema is data**, in malli's vector shape:

```
schema := keyword                      ; a leaf, e.g. :tensor
        | [tag & children]             ; a composite, e.g. [:and :tensor ...]
```

The `tag` (the keyword, or the vector's head) is the dispatch key. Everything else --
`validate`, `explain`, `humanize`, `check` -- is a thin wrapper over one multimethod:

```clojure
(defmulti -problems (fn [schema value] (schema-tag schema)))
;; returns a seq of human-readable problem strings; empty = valid.
```

To add a schema type you add a `defmethod` keyed on its tag. There is no protocol to
implement and no central registry to edit -- just a method. That is the whole extension
mechanism.

### The tags in the minimal core (implemented + green)

| Tag | Validates | Implementation |
|---|---|---|
| `:tensor` | a structurally well-formed tensor map | reuses `jabla.schema/explain-tensor` |
| `:and` | value satisfies **every** child schema | collects problems from each child |
| `:matmul` | a 2-tuple `[a b]` of tensors with matching inner dims | reuses `jabla.schema/matmul-compatible?` |
| `:same-shape` | a 2-tuple `[a b]` of tensors with equal shapes (add/mul) | reuses `jabla.schema/same-shape?` |

`:matmul` and `:same-shape` share a private `pair-problems` helper (a 2-tuple of
well-formed tensors + a shape relation) -- the template for any future binary-op boundary.

`:tensor` is a leaf; `:and` is a composite (shows it nests); `:matmul` is the
interesting one -- a **dependent-shape** check whose "value" is the op's argument tuple.
That is how a shape relation ("matmul's inner dims must match") fits the
validate-one-value contract: the one value is the arg list.

### Public API (mirrors malli on purpose)

```clojure
(jalli/validate :tensor t)          ; => true / false
(jalli/explain  :tensor t)          ; => nil | ["...problem..."]   (data)
(jalli/humanize (jalli/explain ...)); => nil | "problem; problem"  (one string)
(jalli/check    [:matmul] [a b])    ; => [a b], or throws ex-info naming the problems
```

The names and shapes match malli so muscle memory transfers and a later swap is
mechanical (see Migration).

## Relationship to `jabla.schema`

`jabla.schema` is the **predicate layer**: `tensor?`, `explain-tensor`,
`matmul-compatible?`, `same-shape?` -- plain functions, zero deps. jalli is the
**combinator layer** on top: it makes those predicates composable and declarative and
gives them malli's API. The dependent-shape predicates stay in `jabla.schema` as code
(they are code in malli too -- no schema DSL expresses them ergonomically). jalli
neither replaces nor duplicates them; it dispatches to them.

```
jabla.schema  (predicates: tensor?, matmul-compatible?, ...)
      ^
      | reused by leaf/relation methods
      |
jabla.jalli   (schema-as-data + defmulti -problems + validate/explain/humanize/check)
```

## Using it at an op boundary

Two ways, both in use. The low-level form is an inline `check`:

```clojure
(defn matmul [t1 t2] (jalli/check [:matmul] [t1 t2]) ...)   ; clear error, not a segfault
```

The preferred form puts the contract **on the definition** via `defn-checked` (below):

```clojure
(jalli/defn-checked matmul
  "..."                              ; docstring (optional, like defn)
  {:args [:matmul] :ret :tensor}     ; the contract: args-vector schema + return schema
  [t1 t2]
  (->node (matmul-raw t1 t2 false false) :matmul [t1 t2]))
```

All five `jabla.tensor` forwards use `defn-checked` now:

| op | `:args` | `:ret` |
|---|---|---|
| `matmul` | `[:matmul]` (2 rank-2 tensors, inner dims match) | `:tensor` |
| `add` / `mul` | `[:same-shape]` (2 equal-shape tensors) | `:tensor` |
| `gelu` / `relu` | `[:tuple :tensor]` (1 tensor) | `:tensor` |

Kept opt-in by being on the public forwards only -- the `-raw` kernels used in the hot
backward path are NOT checked; it can be flagged or compiled off later. The shape checks
are the high-value ones -- a bad shape otherwise reads past a C++ buffer.

## `defn-checked` -- the function-contract layer

`defn-checked` is jalli's answer to "validate a function's input AND output," the way
plumatic `s/defn :-`, clojure.spec `s/fdef`, and guardrails `>defn` do on the JVM (none
of which load on jank). It's a thin macro -- the only one in jalli -- so the available
jank primitives (defmacro + the data engine) are enough:

```clojure
(jalli/defn-checked name docstring? {:args <schema> :ret <schema>} [params] body...)
```

- `:args` validates the **vector of argument values** -- which is why a *relation* over
  the args (`[:matmul]`, `[:same-shape]`) fits, as does `[:tuple ...]` positionally.
- `:ret` validates the return value.
- Both keys optional; each expands to a `jalli/check` (throws on violation) -- `:args`
  before the body, `:ret` after.

Deliberately minimal: single arity, plain-symbol params (no destructuring / `& rest` /
multi-arity). jabla's ops fit; lift it if an op needs more. clj-kondo lints it as `defn`
(the contract map lands in defn's attr-map slot) via a one-line `:lint-as` in
`.clj-kondo/config.edn`.

## Extending: the next tags

Driven by real op-boundary needs, roughly in order:

- `[:tensor {:dtype :f32 :shape [m n]}]` -- a leaf with **props**: constrain dtype and/or
  an exact/partial shape, not just "is a tensor".
- `[:fn pred]` -- escape hatch: an arbitrary predicate. (malli has this; it is how every
  dependent check ultimately bottoms out.)

## Migration to real malli (someday)

The external-lib path opens only when **both** (a) jank gains protocols/records + the
lexer fix (`private/` tracks the bugs), and (b) malli ships `:jank` reader-conditional
branches (because of the `:jank` branch issue above). Until then jalli stands alone. When
that day comes, the swap is mechanical because the surface already matches:

| jalli | malli |
|---|---|
| `[:and ...]`, `[:tensor]`, `[:tuple ...]` | same vector grammar |
| `validate` / `explain` | `m/validate` / `m/explain` |
| `humanize` | `me/humanize` |
| `-problems` defmethod per tag | `m/-IntoSchema` + `-validator` |
| `defn-checked` `{:args .. :ret ..}` | `[:=>]` schema + `malli.instrument` (or guardrails `>defn`) |

Schemas (the data) would largely carry over; only the engine (defmulti -> protocols) and
the leaf predicates' wiring change. The dependent-shape predicates remain custom code in
either world.

## Status

Landed and green on jank: `src/jabla/jalli.jank` + `test/jabla/jalli_test.jank` (in the
`make test` runner). Tags `:tensor`, `:and`, `:matmul`, `:same-shape`, `:tuple`, plus the
`defn-checked` contract macro. **All five `jabla.tensor` public forwards** (matmul / add /
mul / gelu / relu) declare their input/output contract via `defn-checked`; the throw path
is pinned by `op-boundary-guards` in `tensor_test.jank`. Suite: 63 tests / 160 assertions.
Next: leaf props (`[:tensor {:shape ...}]`) and `[:fn pred]` as new ops need them.
