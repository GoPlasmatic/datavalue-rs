<div align="center">
  <img src="https://avatars.githubusercontent.com/u/207296579?s=200&v=4" alt="Plasmatic Logo" width="120" height="120">

# datavalue-rs
**A bump-allocated JSON value type with a built-in zero-copy parser and `serde_json::Value`-style access.**

  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)
  [![Crates.io](https://img.shields.io/crates/v/datavalue-rs.svg)](https://crates.io/crates/datavalue-rs)
  [![Documentation](https://docs.rs/datavalue-rs/badge.svg)](https://docs.rs/datavalue-rs)

</div>

---

## Quick Example

```rust
use bumpalo::Bump;
use datavalue_rs::DataValue;

let arena = Bump::new();
let v = DataValue::from_str(r#"{"name":"alice","ages":[30,31]}"#, &arena).unwrap();

assert_eq!(v["name"].as_str(), Some("alice"));
assert_eq!(v["ages"][1].as_i64(), Some(31));
assert!(v["missing"].is_null()); // missing key indexes to Null, like serde_json

// Render back to JSON via Display.
println!("{v}");                    // compact: {"name":"alice","ages":[30,31]}
println!("{}", v.pretty());         // two-space indent, like serde_json::to_string_pretty
```

### Building values inline

```rust
use datavalue_rs::{owned_json, OwnedDataValue};

let user = owned_json!({
    "name": "alice",
    "ages": [30, 31],
    "active": true,
    "tags": null,
});

assert_eq!(user["ages"][1].as_i64(), Some(31));
```

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| [datavalue-rs](https://crates.io/crates/datavalue-rs) | Rust library | `cargo add datavalue-rs` |

## Resources

- [Rust API (docs.rs)](https://docs.rs/datavalue-rs)
- [Benchmarks](BENCHMARKS.md) — full cross-library comparison (parse / serialize / access / mutate × twitter / citm / canada)
- [datalogic-rs](https://github.com/GoPlasmatic/datalogic-rs) — JSONLogic engine that consumes this crate

## Key Features

- **Arena-Allocated** — One `Bump` holds the entire value tree; reset between batches for amortized zero-allocation parsing.
- **Zero-Copy Strings** — String literals without escape sequences are borrowed directly from the input source.
- **Native Integer Path** — `NumberValue` distinguishes `Integer(i64)` from `Float(f64)`; integer JSON stays on the integer fast path.
- **`serde_json::Value`-Style Access** — `Index`, `get()`, `as_*`/`is_*`, chained indexing returns `Null` on miss.
- **Owned Counterpart** — `OwnedDataValue` for cases where the value must outlive its arena (caches, return values, global state).
- **Optional `serde` Integration** — `Serialize` for both forms; `DataValueSeed` (DeserializeSeed) for arena targets, direct `Deserialize` for owned.
- **Optional `serde_json::Value` Bridge** — direct conversion to/from `serde_json::Value` for boundary interop (no string round-trip).
- **`Display` + `pretty()`** — `println!("{v}")` emits compact JSON; `v.pretty()` renders the same shape as `serde_json::to_string_pretty`.
- **`owned_json!` Macro** — `serde_json::json!`-style construction for `OwnedDataValue`.
- **Optional `datetime` Extension** — `DateTime` / `Duration` variants backed by `chrono`, mirroring `datalogic-rs`.
- **Optional `tensor` Extension** — `Tensor` variant: an opaque n-dimensional typed buffer (`f32`, `i64`, `bf16`, …) that a JSON tree carries between operators without expanding it into `Array` nodes. Zero-copy typed views, nested-array conversion, and a `{"tensor": …}` wire form. No dependencies.

## Performance

Highlights on the 631 KB `twitter.json` fixture from the serde-rs json
benchmark suite (release build, single thread, criterion):

| Workload                | datavalue-rs | serde_json   | speedup |
|-------------------------|--------------|--------------|---------|
| Parse                   | 1.17 GiB/s   | 0.43 GiB/s   | **2.7×** |
| Walk all status entries | 1.97 µs      | 6.09 µs      | **3.1×** |

Full cross-library numbers (vs. simd-json, sonic-rs, json-rust) across
twitter / citm_catalog / canada and across parse / serialize / access /
mutate workloads live in [BENCHMARKS.md](BENCHMARKS.md). Reproduce with
`cargo bench --bench compare --features serde_json`.

## Owned Counterpart

Use `OwnedDataValue` when a value must escape its arena lifetime — long-lived
caches, function return values, global state. Variants mirror `DataValue`
one-for-one but use `String` / `Vec<…>` / `Vec<(String, …)>` instead of arena
slices.

```rust
use bumpalo::Bump;
use datavalue_rs::{DataValue, OwnedDataValue};

// Parse fast path, then escape the arena.
let arena = Bump::new();
let v = DataValue::from_str(r#"{"x":42}"#, &arena).unwrap();
let owned: OwnedDataValue = v.to_owned();
drop(arena); // arena gone — `owned` keeps living.

assert_eq!(owned["x"].as_i64(), Some(42));

// Or: parse straight into owned form.
let owned2: OwnedDataValue = r#"{"x":42}"#.parse().unwrap();

// Rehydrate into a fresh arena when you need the borrowed shape again.
// `to_arena` copies every string into the arena; `view_in` borrows them
// from `owned2` and allocates only the array/object slices.
let arena2 = Bump::new();
let _copied = owned2.to_arena(&arena2);
let _viewed = owned2.view_in(&arena2); // `owned2` stays borrowed while alive
```

`OwnedDataValue` implements `Serialize` + `Deserialize` directly (no seed
required) since there's no arena lifetime to thread.

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | `Serialize` for both forms; `DataValueSeed` (DeserializeSeed) for arena targets; `Deserialize` for `OwnedDataValue`. |
| `serde_json` | off | Implies `serde`. Bidirectional `From`/`Into` between both value types and `serde_json::Value` (`OwnedDataValue::from_serde_value`, `to_serde_value`, `DataValue::from_serde_value_in`). |
| `datetime` | off | Adds `DateTime(DataDateTime)` / `Duration(DataDuration)` variants (chrono-backed). Mirrors `datalogic-rs`. |
| `tensor` | off | Adds `Tensor(&DataTensor)` / `Tensor(Arc<OwnedDataTensor>)` variants plus `DType`, typed views (`as_slice::<f32>()`), nested `Array` ↔ tensor conversion, and the `{"tensor": …}` JSON form. No dependencies; little-endian targets only. |
| `tensor-half` | off | Extends `tensor` with `f16` / `bf16` as native elements: typed views and nested-array conversion for the two half dtypes, which `tensor` alone carries as bytes only. Adds the `half` dependency, re-exported as `datavalue_rs::half`. |

## Design Notes

- **Strings:** when a JSON string has no escape sequences, the parser borrows
  directly from the input — zero copy. Escaped strings are unescaped into
  the arena.
- **Numbers:** `NumberValue` natively distinguishes `Integer(i64)` from
  `Float(f64)`. Integer-valued JSON numbers stay on the integer fast path
  through arithmetic and access.
- **Objects:** `&'a [(&'a str, DataValue<'a>)]`. Insertion order is
  preserved; lookup is a linear scan, which beats a hash map for the
  typical small object sizes seen in JSON.
- **No coercion:** `as_i64`, `as_str`, etc. return `None` when the variant
  doesn't match. There is no JSONLogic-style truthiness or cross-type
  coercion here; that belongs in the consumer crate.
- **DateTime:** JSON has no native datetime type, so the parser does not
  produce `DateTime` / `Duration` variants. Consumers upgrade strings at
  the operator boundary via `DataDateTime::parse`. Serialization back to
  JSON emits an ISO 8601 string or `"1d:2h:3m:4s"` duration string.
- **Tensor:** a `DataTensor` is a dtype, a shape, and one row-major,
  contiguous, native-endian byte buffer aligned for its dtype — one
  safetensors entry, one ONNX Runtime input, one numpy buffer, with zero
  copies. The parser never produces it. Consumers build one from a typed
  slice (`DataTensor::from_slice`, zero-copy), from nested JSON arrays
  (`from_nested_in`, a typed decode into a declared dtype that refuses
  anything that does not fit), from the wire form (`from_json_value_in`), or
  by filling an aligned buffer in place (`zeroed_bytes_in` + `from_bytes`,
  the single-pass route for code that assembles bytes itself);
  `from_bytes_in` copies when a buffer's alignment is not yours to choose. It renders as
  `{"tensor":{"dtype":"f32","shape":[2,3],"data":"<base64>"}}` — base64 in
  every serde format, binary ones included — and the `Deserialize` impl and
  the parser leave that form as an `Object` until a consumer asks for the
  upgrade. A shape may have at most `MAX_RANK` (256) dimensions, so decoding
  an untrusted payload cannot be turned into an unbounded recursion. `get` / `len` / `members` treat a
  tensor as opaque. Equality is bytewise. No arithmetic, no dtype
  conversion: those belong in the consumer.

## Status

`0.3` — public API may shift before `1.0`. Requires Rust 1.98 or newer.
Built to back hot paths in
`datalogic-rs` and other Plasmatic crates.

## Contributing

Contributions are welcome. Fork the repo, add tests for any new behavior,
and open a PR.

## About Plasmatic

Created by [Plasmatic](https://github.com/GoPlasmatic), building open-source
tools for financial infrastructure and data processing.

## License

Licensed under Apache 2.0. See [LICENSE](LICENSE) for details.
