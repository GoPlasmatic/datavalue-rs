# datavalue

Bump-allocated JSON value type with a built-in zero-copy parser and a
`serde_json::Value`-style access API.

`DataValue<'a>` mirrors the shape of `serde_json::Value`, but every
composite payload — string bytes, array elements, object pairs — lives in a
[`bumpalo::Bump`] arena. Designed for hot paths where per-value heap
allocation is the bottleneck and the same arena handles many values
between resets.

## At a glance

```rust
use bumpalo::Bump;
use datavalue::DataValue;

let arena = Bump::new();
let v = DataValue::from_str(r#"{"name":"alice","ages":[30,31]}"#, &arena).unwrap();

assert_eq!(v["name"].as_str(), Some("alice"));
assert_eq!(v["ages"][1].as_i64(), Some(31));
assert!(v["missing"].is_null()); // missing key indexes to Null, like serde_json
```

## Performance

Measured on the 631 KB `twitter.json` fixture from the serde-rs json
benchmark suite (release build, single thread, criterion):

| Workload                | datavalue   | serde_json   | speedup |
|-------------------------|-------------|--------------|---------|
| Parse (fresh arena)     | 1.18 GiB/s  | 0.46 GiB/s   | **2.6×** |
| Parse (reused arena)    | 1.20 GiB/s  | 0.46 GiB/s   | **2.6×** |
| Walk all status entries | 1.93 µs     | 6.06 µs      | **3.1×** |

Reproduce with `cargo bench --bench twitter`.

## Owned counterpart

Use [`OwnedDataValue`] when a value must outlive its arena (long-lived
caches, function return values, global state). Variants mirror `DataValue`
1:1 but use `String` / `Vec<…>` / `Vec<(String, …)>` instead of arena
slices.

```rust
use datavalue::{DataValue, OwnedDataValue};
use bumpalo::Bump;

// Parse fast path, then escape the arena.
let arena = Bump::new();
let v = DataValue::from_str(r#"{"x":42}"#, &arena).unwrap();
let owned: OwnedDataValue = v.to_owned();
drop(arena); // arena gone — `owned` keeps living.

assert_eq!(owned["x"].as_i64(), Some(42));

// Or: parse straight into owned form.
let owned2: OwnedDataValue = r#"{"x":42}"#.parse().unwrap();

// Rehydrate back into an arena when you need the borrowed shape again.
let arena2 = Bump::new();
let borrowed = owned2.to_arena(&arena2);
```

`OwnedDataValue` implements `Serialize` + `Deserialize` directly (no seed
required) since there's no arena lifetime to thread.

## Design notes

- **Strings**: when a string has no escape sequences, the parser borrows
  directly from the input — zero copy. Escaped strings are unescaped into
  the arena.
- **Numbers**: `NumberValue` natively distinguishes `Integer(i64)` from
  `Float(f64)`. Integer-valued JSON numbers stay on the integer fast path
  through arithmetic and access.
- **Objects**: `&'a [(&'a str, DataValue<'a>)]`. Insertion order is
  preserved; lookup is a linear scan, which beats a hash map for the
  typical small object sizes seen in JSON.
- **No coercion**: `as_i64`, `as_str`, etc. return `None` when the variant
  doesn't match. There is no JSONLogic-style truthiness or cross-type
  coercion here; that belongs in the consumer crate.

## Features

- `serde` — `impl Serialize for DataValue` plus a `DataValueSeed`
  `DeserializeSeed` adapter so `DataValue` can be deserialized from any
  serde data format (msgpack, flexbuffers, `serde_json::Deserializer`,
  etc.). For JSON specifically, prefer the built-in parser — it's faster.
- `serde_json` — implies `serde`. Pulls in `serde_json` for
  boundary-bridge use.
- `datetime` — adds `DataValue::DateTime(DataDateTime)` and
  `DataValue::Duration(DataDuration)` variants (chrono-backed). Mirrors the
  shape exposed by `datalogic-rs`. JSON has no native datetime type, so the
  parser does not produce these — consumers upgrade strings via
  `DataDateTime::parse` at the operator boundary. Serialization back to
  JSON emits an ISO 8601 datetime string or `"1d:2h:3m:4s"` duration
  string.

## Status

`0.1` — public API may shift before `1.0`. Built to back hot paths in
`datalogic-rs` and other Plasmatic crates.
