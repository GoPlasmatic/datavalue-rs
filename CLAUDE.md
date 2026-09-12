# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Default build / test / lint
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check

# Feature combos must both pass — CI runs both
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings

# Single test
cargo test --all-features parser::tests::string_escapes

# Cross-library benchmarks (twitter, citm_catalog, canada fixtures live in
# benches/fixtures/). Compares against serde_json, simd-json, sonic-rs, and
# json-rust across parse / serialize / access / mutate workloads.
cargo bench --bench compare --features serde_json
cargo bench --bench compare --features serde_json -- --quick   # short run
```

The crate is published as `datavalue-rs`; the library name auto-converts to `datavalue_rs`. Imports must use the underscored form (`use datavalue_rs::DataValue;`). Doctests, benches, and README all assume that spelling.

## Architecture

### Two parallel value types

`DataValue<'a>` (arena-bound, `Copy`) and `OwnedDataValue` (heap-owned) are intentionally kept variant-for-variant identical:

| Variant | `DataValue<'a>` | `OwnedDataValue` |
|---|---|---|
| String | `&'a str` | `String` |
| Array | `&'a [DataValue<'a>]` | `Vec<OwnedDataValue>` |
| Object | `&'a [(&'a str, DataValue<'a>)]` | `Vec<(String, OwnedDataValue)>` |
| DateTime / Duration (feature-gated) | inline `DataDateTime` / `DataDuration` | same |
| Tensor (feature-gated `tensor`) | `&'a DataTensor<'a>` (behind a reference: 40-byte header, enum stays 24) | `Arc<OwnedDataTensor>` (the one shared-ownership payload: immutable, potentially huge, clone is a refcount bump) |

**Any change to one type must be mirrored in the other**: add a variant → add to both enums; add an accessor → add to both impls; add a feature-gated branch → gate both. The same applies to `Serialize` impls in `ser.rs` and the `to_owned()` / `to_arena()` conversion methods. The conversion pair is what holds them in sync at runtime; the access surface is what holds them in sync ergonomically.

`DataValue` is `#[derive(Copy)]` — every variant payload must remain `Copy`. `chrono::DateTime<Utc>` and `chrono::Duration` are `Copy`, which is why `DataDateTime` / `DataDuration` are inline rather than boxed.

`const` assertions in `value.rs` / `owned.rs` pin the enums at 24 / 32 bytes on 64-bit. A new variant whose payload is bigger than 16 bytes must go behind a reference (arena side) or an `Arc` / `Box` (owned side), as `Tensor` does.

**Extension-slot trigger.** `Tensor` is the second foreign type after `DateTime`. If a third one is proposed (a `Decimal`, `BigInt`, `Uuid`, `Bytes`), design a generic extension slot (`DataValue::Ext(&'a dyn ExtValue)` / `OwnedDataValue::Ext(Arc<dyn ExtValue>)`) before adding a fourth concrete variant — see `proposal.md` §3.14 for the evaluated trade-offs.

### Tensor (`src/tensor.rs`, feature `tensor`)

Fields are private; three invariants hold for every tensor and make the single `unsafe` typed view (`as_slice::<T>()`) sound: byte length equals `dtype.byte_len(numel)`, the buffer is aligned to `dtype.align_of()`, and a `Bool` payload is all `0`/`1`. Every constructor enforces them; keep it that way. Separately from those, `MAX_RANK` (256, matching the parser's `MAX_DEPTH`) bounds `shape.len()` — not a soundness invariant but a resource limit, checked in `expected_byte_len`, which every construction path reaches before allocating and before any recursion, so a wire payload cannot drive `fill_rec` / `emit_rec` past the stack budget. Memory is native-endian, the wire form is little-endian, and big-endian targets get a `compile_error!`.

`from_nested_in` / `to_nested_in` are the one place the crate decodes JSON leaves into scalars. The rule that keeps this on the right side of the no-coercion policy: the caller declares the dtype; an element that does not fit (wrong JSON type, non-whole float for an integer dtype, out of range, or overflow to a non-finite float) is a `TensorError::Element`, never a truncation, saturation, or manufactured `±inf`. That follows `as_i64()`, which returns `None` rather than altering a value. No dtype inference, no numeric strings, no truthiness.

The `{"tensor": ...}` decoder (`from_json_value_in`) is strict: an unknown key inside `tensor` is an error, so a future `strides` field cannot be silently misread by an old decoder. The parser and `Deserialize` never produce `Tensor`; only that explicit call does.

### Parser (`src/parser.rs`)

Hand-rolled recursive-descent over `&[u8]`, single linear scan, no backtracking. Two paths matter:

- **Strings**: scan for terminator; if no `\` is seen, return a `&str` slice into the original input (zero-copy). The slow `parse_string_with_escapes` path only runs when an escape is encountered, copying byte-by-byte into a `bumpalo::Vec` and emitting `unsafe core::str::from_utf8_unchecked` over the result. This is sound because the input is `&str` (already valid UTF-8) and the unescape path only emits valid UTF-8 byte sequences (escapes are ASCII, `\u` paths use `char::encode_utf8`).
- **Numbers**: integer fast path parses to `i64`; only on overflow or seeing `.`/`e`/`E` do we fall through to `f64`. The integer path is preserved through `NumberValue::Integer` so downstream arithmetic stays in i64.

`MAX_DEPTH` (256) caps recursion to keep the stack bounded on adversarial input.

The parser **never produces** `DateTime` / `Duration` / `Tensor` variants. JSON has no native datetime or tensor; consumer crates (e.g. `datalogic-rs`) upgrade `String` → `DateTime` (via `DataDateTime::parse`) and `Object` → `Tensor` (via `DataTensor::from_json_value_in`) at the operator boundary.

### Index trait dispatch

`ValueIndex` (for `DataValue`) and `OwnedValueIndex` (for `OwnedDataValue`) are sealed traits implemented for `str`, `String`, `usize`, and `&T`. Both expose `index_into` (returns `Option`) and `index_into_or_null` (returns `&Self` falling back to a static `Null`). The `Null` fallback is what makes `v["missing"]["also_missing"]` chain without panicking — the `Index<I>` impl uses `index_into_or_null`. Object lookup is a linear scan; preferred over hash maps since most JSON objects have ≤16 keys.

### Serde shape (feature-gated, `src/ser.rs`)

The arena-bound side cannot implement `Deserialize` directly because deserialization needs a `&Bump` to allocate into. The pattern is:

- `DataValue` gets `impl Serialize` only.
- `DataValueSeed<'a> { arena: &'a Bump }` carries the arena via `DeserializeSeed`. Use this when plugging into existing serde flows (msgpack/flexbuffers/`serde_json::Deserializer`).
- `OwnedDataValue` gets both `Serialize` and `Deserialize` directly — no seed, since there's no arena lifetime to thread.
- A tensor serializes as the tagged map with `data` base64 in **every** format, binary ones included. No `visit_bytes` exists on either visitor, so a byte string would be write-only; uniform base64 keeps the lossy round-trip recoverable (`Object` back in, `DataTensor::from_json_value` rebuilds it). See `proposal.md` §3.7.

For JSON specifically, `DataValue::from_str` (the hand-rolled parser) is faster than going through `DataValueSeed` + `serde_json::Deserializer`. The seed is for non-JSON formats and existing serde pipelines.

### Equality

Object equality is **order-insensitive** (`PartialEq` matches by key set, not key order) — this matches `serde_json::Value` semantics and is shared between `DataValue` and `OwnedDataValue`. Don't replace this with a slice equality shortcut.

### What this crate is NOT

- **No coercion, no truthiness, no cross-type conversions**. `as_i64()` returns `None` for a string `"42"`. Coercion belongs in consumer crates (e.g. `datalogic-rs`). Pull requests adding `is_truthy`, `coerce_to_*`, or string-number conversions should be redirected.
- **No mutation**. Everything is read-mostly; the arena is the unit of mutation (reset between batches).

## CI

`.github/workflows/ci.yml` runs fmt + clippy + tests for **both** default features and `--all-features`. Both must pass. `.github/workflows/release.yml` validates that the git tag matches the `Cargo.toml` version, runs the same gate, then publishes to crates.io with a "skip if already published" guard.
