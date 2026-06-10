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

```clojure
(defn matmul [a b]
  (jalli/check [:matmul] [a b])      ; clear error instead of a cblas segfault
  ...)
```

Keep it opt-in: on at the API edges and while bringing an op up; it can be flagged or
compiled off the hot path later. The shape checks are the high-value ones -- a bad shape
currently reads past a C++ buffer.

## Extending: the next tags

Driven by real op-boundary needs, roughly in order:

- `[:tensor {:dtype :f32 :shape [m n]}]` -- a leaf with **props**: constrain dtype and/or
  an exact/partial shape, not just "is a tensor".
- `:same-shape` -- a 2-tuple `[a b]` with equal shapes (the elementwise-op precondition;
  reuses `jabla.schema/same-shape?`). The `add` / `mul` analogue of `:matmul`.
- `[:tuple s1 s2 ...]` -- positional: each element of a vector value matches its schema.
  Generalizes the ad-hoc 2-tuple handling in `:matmul` / `:same-shape`.
- `[:fn pred]` -- escape hatch: an arbitrary predicate. (malli has this; it is how every
  dependent check ultimately bottoms out.)

## Migration to real malli (someday)

The external-lib path opens only when **both** (a) jank gains protocols/records + the
lexer fix (`private/` tracks the bugs), and (b) malli ships `:jank` reader-conditional
branches (because of the `:jank` branch issue above). Until then jalli stands alone. When
that day comes, the swap is mechanical because the surface already matches:

| jalli | malli |
|---|---|
| `[:and ...]`, `[:tensor]`, `[:fn p]` | same vector grammar |
| `validate` / `explain` | `m/validate` / `m/explain` |
| `humanize` | `me/humanize` |
| `-problems` defmethod per tag | `m/-IntoSchema` + `-validator` |

Schemas (the data) would largely carry over; only the engine (defmulti -> protocols) and
the leaf predicates' wiring change. The dependent-shape predicates remain custom code in
either world.

## Status

Minimal working core landed and green on jank: `src/jabla/jalli.jank` +
`test/jabla/jalli_test.jank` (in the `make test` runner). Tags `:tensor`, `:and`,
`:matmul`. Not yet wired into the `jabla.tensor` op boundaries -- that adoption is the
next step (see "Using it at an op boundary").
