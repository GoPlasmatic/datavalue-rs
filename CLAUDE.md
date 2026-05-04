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

**Any change to one type must be mirrored in the other**: add a variant → add to both enums; add an accessor → add to both impls; add a feature-gated branch → gate both. The same applies to `Serialize` impls in `ser.rs` and the `to_owned()` / `to_arena()` conversion methods. The conversion pair is what holds them in sync at runtime; the access surface is what holds them in sync ergonomically.

`DataValue` is `#[derive(Copy)]` — every variant payload must remain `Copy`. `chrono::DateTime<Utc>` and `chrono::Duration` are `Copy`, which is why `DataDateTime` / `DataDuration` are inline rather than boxed.

### Parser (`src/parser.rs`)

Hand-rolled recursive-descent over `&[u8]`, single linear scan, no backtracking. Two paths matter:

- **Strings**: scan for terminator; if no `\` is seen, return a `&str` slice into the original input (zero-copy). The slow `parse_string_with_escapes` path only runs when an escape is encountered, copying byte-by-byte into a `bumpalo::Vec` and emitting `unsafe core::str::from_utf8_unchecked` over the result. This is sound because the input is `&str` (already valid UTF-8) and the unescape path only emits valid UTF-8 byte sequences (escapes are ASCII, `\u` paths use `char::encode_utf8`).
- **Numbers**: integer fast path parses to `i64`; only on overflow or seeing `.`/`e`/`E` do we fall through to `f64`. The integer path is preserved through `NumberValue::Integer` so downstream arithmetic stays in i64.

`MAX_DEPTH` (256) caps recursion to keep the stack bounded on adversarial input.

The parser **never produces** `DateTime` / `Duration` variants. JSON has no native datetime; consumer crates (e.g. `datalogic-rs`) upgrade `String` → `DateTime` at the operator boundary.

### Index trait dispatch

`ValueIndex` (for `DataValue`) and `OwnedValueIndex` (for `OwnedDataValue`) are sealed traits implemented for `str`, `String`, `usize`, and `&T`. Both expose `index_into` (returns `Option`) and `index_into_or_null` (returns `&Self` falling back to a static `Null`). The `Null` fallback is what makes `v["missing"]["also_missing"]` chain without panicking — the `Index<I>` impl uses `index_into_or_null`. Object lookup is a linear scan; preferred over hash maps since most JSON objects have ≤16 keys.

### Serde shape (feature-gated, `src/ser.rs`)

The arena-bound side cannot implement `Deserialize` directly because deserialization needs a `&Bump` to allocate into. The pattern is:

- `DataValue` gets `impl Serialize` only.
- `DataValueSeed<'a> { arena: &'a Bump }` carries the arena via `DeserializeSeed`. Use this when plugging into existing serde flows (msgpack/flexbuffers/`serde_json::Deserializer`).
- `OwnedDataValue` gets both `Serialize` and `Deserialize` directly — no seed, since there's no arena lifetime to thread.

For JSON specifically, `DataValue::from_str` (the hand-rolled parser) is faster than going through `DataValueSeed` + `serde_json::Deserializer`. The seed is for non-JSON formats and existing serde pipelines.

### Equality

Object equality is **order-insensitive** (`PartialEq` matches by key set, not key order) — this matches `serde_json::Value` semantics and is shared between `DataValue` and `OwnedDataValue`. Don't replace this with a slice equality shortcut.

### What this crate is NOT

- **No coercion, no truthiness, no cross-type conversions**. `as_i64()` returns `None` for a string `"42"`. Coercion belongs in consumer crates (e.g. `datalogic-rs`). Pull requests adding `is_truthy`, `coerce_to_*`, or string-number conversions should be redirected.
- **No mutation**. Everything is read-mostly; the arena is the unit of mutation (reset between batches).

## CI

`.github/workflows/ci.yml` runs fmt + clippy + tests for **both** default features and `--all-features`. Both must pass. `.github/workflows/release.yml` validates that the git tag matches the `Cargo.toml` version, runs the same gate, then publishes to crates.io with a "skip if already published" guard.
