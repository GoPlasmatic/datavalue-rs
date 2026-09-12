# Proposal: `tensor` feature — an opaque n-dimensional typed buffer (GH #1)

Status: **implemented on the working tree (2026-09-12), uncommitted.** All
seven plan steps are done; `cargo test` / `clippy -D warnings` / `fmt --check`
pass under default, `--features tensor`, and `--all-features`.
Scope: `datavalue-rs` only. Consumer work (`datalogic-rs` operators) is listed
in §6 as follow-ups.

---

## 0. TL;DR

The issue's shape is right and is kept: feature-gated variant, mirrored on
both enums, parser never produces it, consumers create it at an operator
boundary — the `datetime` precedent.

Reviewing it against the code changed seven things. All are cheap now and
expensive after the byte layout and type signatures become public API.

| # | Issue proposes | Decision | Why |
|---|---|---|---|
| 1 | `OwnedDataValue::Tensor(OwnedTensor)` inline | `Tensor(Arc<OwnedDataTensor>)` | Inline payload is 56 bytes and **doubles `OwnedDataValue` from 32 to 64 bytes** (measured). `Arc` keeps 32 and makes clone O(1) for buffers that are immutable and potentially huge. |
| 2 | `pub dtype / pub shape / pub data` | Private fields, validated constructors, accessors | With public fields the length invariant is unenforceable and the representation (contiguous, no strides) is frozen forever. |
| 3 | Alignment unspecified | Alignment is an invariant; every constructor guarantees it | Without it consumers cannot get `&[f32]` from `&[u8]` without a copy or undefined behaviour. This is what makes "zero-copy into ONNX Runtime" true. |
| 4 | "datavalue never interprets elements" | Typed **views** (`as_slice::<T>()`); still no conversions or arithmetic | A view is a reinterpretation, not a coercion. Every consumer would otherwise write the same `unsafe` cast. |
| 5 | "little-endian native bytes" | Memory native-endian; wire little-endian; `compile_error!` on big-endian | "little-endian native" is contradictory on big-endian hosts. Fail loudly rather than ship untested swap code. |
| 6 | Round-trip "not claimed" | Parser and `Deserialize` untouched; explicit boundary decoder added | Mirrors `DataDateTime::parse`. The "read back" operator needs it; encoder and decoder must live together so they cannot drift. |
| 7 | `data` always base64 in `Serialize` | base64 in every format (revised — see §3.7) | The `is_human_readable()` split shipped first, then lost its argument: nothing in the crate can read a byte string back, so binary documents did not round-trip at all. |

Additionally in v1, by decision: nested `Array` ↔ tensor conversion
(§4.8), with an explicit rule for where it sits relative to the crate's
no-coercion policy (§3.15).

Smaller items: custom `Debug` (the derive would dump megabytes),
`DType::byte_len(numel)` so packed sub-byte dtypes can be added later
without redefining the invariant, `0.3.0` release, `OwnedDataTensor` naming.

A generic extension slot (`DataValue::Ext(&'a dyn ExtValue)`) instead of a
concrete variant was evaluated and rejected for now (§3.14); the trigger for
revisiting it is recorded in `CLAUDE.md` as part of this work.

---

## 1. What the issue asks for

- Feature `tensor`, default off, no new mandatory dependency.
- `DType` (non-exhaustive: `Bool, I8..U64, F16, BF16, F32, F64`).
- `DataTensor<'a> { dtype, shape: &'a [usize], data: &'a [u8] }` and an owned
  twin; row-major, contiguous, invariant `data.len() == numel * size_of`.
- `DataValue::Tensor(&'a DataTensor<'a>)`, `OwnedDataValue::Tensor(OwnedTensor)`.
- `is_tensor / as_tensor / tensor(..)`, `to_owned / to_arena` arms.
- `get / len / is_empty / members / entries` treat it as opaque.
- Equality: structural (dtype, shape, bytes).
- Rendering: `{"tensor":{"dtype":"f32","shape":[2,3],"data":"<base64>"}}`.
- Parser never produces it; `Deserialize` does not recognise the tagged form.
- Out of scope: operators, arithmetic, element indexing, `ndarray` /
  `safetensors` / `half` deps, fp8 / int4, strides.

## 2. Facts established from the code

Measured with a scratch crate against the current tree (64-bit):

| Type | default features | `--features datetime` |
|---|---|---|
| `DataValue<'_>` | 24 bytes | 24 bytes |
| `OwnedDataValue` | 32 bytes | 32 bytes |
| `DataDateTime` | — | 20 bytes, align 4 |
| `DataDuration` | — | 16 bytes, align 8 |

The `datetime` precedent touches exactly these places, and `tensor` must
touch the same set:

- `Cargo.toml` feature, `lib.rs` module + re-exports.
- `value.rs` / `owned.rs`: variant, `is_*`, `as_*`, constructor, `PartialEq`
  arm, `to_owned` / `to_arena` arms.
- `emit.rs`: compact and pretty arms, for both value types (four match arms).
- `ser.rs`: `Serialize` arms for both types.
- `serde_json_bridge.rs`: `to_serde_value` arms for both types.
- README feature table + design note, `CLAUDE.md` variant table.

There is no existing size assertion in the crate; the issue's acceptance
criterion adds one.

`bumpalo` 3.20 (pinned) exposes `Bump::alloc_layout(Layout) -> NonNull<u8>`
and `alloc_slice_copy<T>`, which are the two aligned-allocation paths the
design relies on.

Consumer footprint: `datalogic-rs` (`crates/datalogic-rs`, plus the Python
binding) has **16 match arms across 14 files** that name the `Duration`
variant. Each will need a `Tensor` arm when it enables the feature. It pins
`datavalue-rs = "0.2.3"` (caret), so a `0.3.0` release does not reach it until
it opts in.

No ONNX / tensor code exists in the sibling repos yet; the adapter the issue
mentions is not local. The data-path answers in §8 stand in for it.

---

## 3. Challenging the requirement

Each item: what the issue says, what is wrong or missing, what was decided.

### 3.1 The owned variant doubles the owned enum

`OwnedTensor { dtype: DType, shape: Vec<usize>, data: Vec<u8> }` is
1 + 7 (pad) + 24 + 24 = 56 bytes. `OwnedDataValue`'s largest payload today is
24 bytes (`Vec` / `String`), so the enum is 32. Inline, it becomes 64 —
every `Vec<OwnedDataValue>` in every consumer that enables the feature pays
double, whether or not it ever holds a tensor. The issue's acceptance
criterion ("the enum did not grow") only mentions `DataValue`.

**Decision.** `OwnedDataValue::Tensor(Arc<OwnedDataTensor>)`.

- 8-byte payload, enum stays 32; both sizes asserted in a `const` block.
- `Arc` rather than `Box`: the crate has no mutation API, tensors are
  immutable once built, and they are the one payload where `clone()` can be
  megabytes. `datalogic-rs` clones owned literals into rule caches (CSE,
  pre-literalised nodes); an `Arc` makes that a refcount bump. The choice is
  visible in the variant's public type, so it is made now.
- `Tensor` is therefore the one payload with shared ownership on the owned
  side. Recorded in `CLAUDE.md`.

### 3.2 Public fields make the invariant a suggestion

With `pub data: &'a [u8]`, any caller can build a tensor whose byte length
disagrees with its shape, and "open decision 2" (validate in `new`) is moot
because `new` can be bypassed. Public fields also freeze the representation:
adding strides, a device tag, or changing the owned storage later is a
breaking change.

**Decision.** Private fields on both structs. Construction through validated
constructors returning `Result<_, TensorError>`; read access through
`dtype()`, `shape()`, `ndim()`, `numel()`, `data()`.

### 3.3 Alignment is unspecified, which breaks the zero-copy claim

The justification for native bytes is "one ONNX Runtime input, one numpy
buffer, zero copies". ONNX Runtime (`ort`) takes `&[f32]`; numpy wants an
aligned buffer for typed views. Turning `&[u8]` into `&[f32]` is only sound
if the pointer is 4-byte aligned. Nothing in the issue guarantees it:
`arena.alloc_slice_copy::<u8>()` returns 1-aligned memory, and `Vec<u8>` from
the global allocator is not contractually aligned either.

**Decision.** Alignment is part of the invariant:
`data.as_ptr() % dtype.align_of() == 0`, enforced at every construction path
(§4.3). Guaranteed alignment is `dtype.align_of()` (at most 8). Higher
alignment for SIMD kernels is not promised; consumers who need it own the
allocation and use the zero-copy wrap.

### 3.4 "Only carries bytes" pushes the same `unsafe` into every consumer

The issue is right about *conversions* (f16 → f32, i64 → f32) and
*arithmetic*: consumer concerns. But a typed **view** — `&[u8]` seen as
`&[f32]` — has no semantics, exactly like `as_str()` on a `String`. Without
it, `datalogic-rs`, the ONNX adapter and the Python binding each write the
same `unsafe` `from_raw_parts` with the same alignment argument, and only
datavalue can make that argument true.

**Decision.** A sealed `Element` trait with `const DTYPE: DType`, implemented
for `bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64`, and
`as_slice::<T: Element>()` returning `None` on dtype mismatch. One `unsafe`
block in the crate, with the soundness argument next to the invariants.

- `bool`: every bit pattern is valid for ints and floats, not for `bool`.
  Constructors validate that a `Bool` payload is all `0`/`1` bytes (O(n),
  `Bool` only, one sequential read; documented).
- `usize` / `isize` are deliberately **not** `Element`: their width is
  platform-dependent and there is no matching `DType`.
- `F16` / `BF16`: no `Element` impl in the base feature (no `half`
  dependency). Consumers read `data()` and cast to `&[u16]` themselves
  (alignment 2 is guaranteed). A later `tensor-half` feature adds impls for
  `half::f16` / `half::bf16`, purely additive.
- Still **no** cross-dtype `to_vec`, `to_f64`, or arithmetic.

### 3.5 "Little-endian native bytes" is two different things

On a big-endian target native bytes are not little-endian. The wire format
must be fixed (little-endian, so it is one safetensors / numpy buffer); the
in-memory form must be native for typed views to be zero-cost.

**Decision.** Memory is native-endian, wire is little-endian. v1 emits
`compile_error!` under `#[cfg(all(feature = "tensor", target_endian =
"big"))]`. No Plasmatic target (x86-64, aarch64, wasm32, the Python / Node /
C bindings) is big-endian. The lift path is a per-dtype byte-swap in the
encoder / decoder with no API change.

### 3.6 No way back from the tagged JSON form

The issue disclaims round-tripping, citing `DateTime`. The analogy is
incomplete: for `DateTime` the way back is `DataDateTime::parse`, one call at
the operator boundary. For tensors the way back is "check the `tensor` key,
look up the dtype, decode base64, validate the length, allocate aligned". If
datavalue does not provide that, the promised "read back" operator
re-implements it with its own base64 decoder in another crate.

**Decision.** Parser and `Deserialize` stay exactly as the issue says. An
explicit boundary decoder is added (§4.6). A user document containing a
`tensor` key is unaffected: nothing upgrades it unless a consumer asks.
Precedent: datalogic's own `{"datetime": ..}` / `{"timestamp": ..}`
sentinel objects.

### 3.7 base64 in every serde format is wrong for binary formats

The `serde` feature exists for non-JSON pipelines (`ser.rs` names msgpack
and flexbuffers). Those have native byte strings; base64 there is 33% larger
plus a decode step.

**Decision (revised).** Same tagged map for every format, `data` base64 in
all of them.

The `is_human_readable()` split was implemented first and then reversed. It
weighed size against nothing, because the other half of the round trip does
not exist: neither `DataValueSeed` nor `Deserialize for OwnedDataValue`
implements `visit_bytes`, so a msgpack document containing a tensor came
back as a hard `invalid type: bytes` — worse than the documented lossy
`Object`. Adding the visitor arm is not the fix either. The honest target
for a byte string is a `Bytes` variant, and that is a third foreign type
after `DateTime` and `Tensor`, which §3.14 defers to an extension slot;
mapping bytes to a `String` or an array of numbers is the coercion this
crate exists to refuse.

Encoding uniformly buys a property the split could not have: the lossy
round trip is *recoverable*. A tensor deserializes as a plain `Object` whose
`data` is base64, and `DataTensor::from_json_value` rebuilds it — in msgpack
exactly as in JSON. The 33% is the price of that, and a consumer that cannot
pay it is better served by a byte-string extension slot than by a wire form
only the encoder understands.

### 3.8 `#[derive(Debug)]` would print the whole buffer

`DataValue` derives `Debug`, which delegates to the payload. A 100 MB tensor
in a log line, a test failure, or `datalogic-rs`'s CSE pass (which today
hashes `format!("{d:?}")` for datetimes) is a footgun.

**Decision.** Manual, lossy `Debug`:
`DataTensor { dtype: f32, shape: [2, 3], bytes: 24 }`.

### 3.9 Equality: bytewise, with the float consequence documented

Structural `memcmp` equality is right for an opaque buffer. It differs from
`NumberValue` float equality: identical `NaN` bit patterns are equal, `0.0`
and `-0.0` are not. That is what safetensors and numpy users expect from a
byte buffer, and it goes in the docs. A tensor never compares equal to an
equivalent nested `Array`.

### 3.10 Sub-byte dtypes will break a `size_of`-based invariant

int4 is 4 bits; an invariant written as `data.len() == numel * size_of()`
cannot express it.

**Decision.** `DType::byte_len(self, numel) -> Option<usize>` (checked) is
the single primitive the invariant is defined in terms of. For every v1
dtype it equals `numel * size_of`. `size_of()` / `align_of()` remain as
conveniences.

### 3.11 Edge cases the issue does not pin down

- **Zero-dimensional** (`shape = []`, ONNX scalars): allowed, `numel = 1`.
- **Zero-size dimensions** (`[0, 3]`): allowed, `numel = 0`, empty `data`,
  wire `"data": ""`.
- **Shape overflow**: `product()` can overflow `usize` on adversarial input
  (the decoder is reachable from user JSON). `checked_mul`; `ShapeOverflow`.
- **`len()` / `is_empty()`**: `None`. Returning `shape[0]` was rejected — it
  would make a tensor look like an array to generic code that then calls
  `get(0)` and receives `None`.

### 3.12 Naming

`OwnedTensor` breaks the `X` / `OwnedX` pairing. **Decision:**
`DataTensor<'a>` / `OwnedDataTensor`.

### 3.13 Versioning

Adding a variant to a public enum without `#[non_exhaustive]` breaks
exhaustive matchers. It is feature-gated, but Cargo feature unification means
a crate can be opted in by a neighbour. **Decision:** release as **0.3.0**.
`datalogic-rs` pins `0.2.3`, so it upgrades deliberately with its arms.

### 3.14 The strategic alternative: a generic extension slot

"The value enum is the contract every consumer evaluates over" cuts both
ways. The next foreign type after tensor is predictable for a
financial-infrastructure company: an arbitrary-precision `Decimal`; then
`BigInt`, `Uuid`, `Bytes`. Each is a datavalue release, a variant on both
enums, four emitter arms, two serde arms, and a `0.x` bump.

The alternative is one open slot:

```rust
DataValue::Ext(&'a dyn ExtValue)          // 16 bytes, enum stays 24
OwnedDataValue::Ext(Arc<dyn ExtValue>)    // 16 bytes, enum stays 32

pub trait ExtValue: Any + Debug + Send + Sync {
    fn type_name(&self) -> &'static str;
    fn ext_eq(&self, other: &dyn ExtValue) -> bool;
    fn write_json(&self, out: &mut dyn Write) -> io::Result<()>;
    fn to_owned_ext(&self) -> Arc<dyn ExtValue>;
}
```

**Decision: concrete `Tensor` now.** The slot trades the crate's core
properties (concrete, `Copy`, statically matched, no dynamic dispatch,
"variant-for-variant identical") for openness nothing needs yet; two
consumers would have to agree out-of-band on the concrete type behind
`Ext`, whereas today the enum *is* the agreement. `CLAUDE.md` gains a note:
if a third foreign type is proposed, design the extension slot before adding
a fourth variant.

### 3.15 Nested-array conversion and the no-coercion rule

The issue leaves "make" and "read back" to consumers. Both are, at their
core, one function pair — nested `Array` → typed buffer with shape inference,
and the reverse — that every consumer would write, and that touches the
crate's no-coercion policy. Decision (§8): it ships in v1, under this rule:

> Conversion is a **typed decode into a caller-declared dtype**, with an
> explicit error for anything that does not fit. It is not coercion: there is
> no string → number, no truthiness, no dtype inference, no silent
> truncation of integers.

Concretely: the caller always passes the `DType`; JSON `Bool` is accepted
only for `Bool`; JSON `Number` is accepted for numeric dtypes with the rules
in §4.8; anything else is an error naming the offending element.

---

## 4. Revised design

### 4.1 Types (`src/tensor.rs`, `#[cfg(feature = "tensor")]`)

```rust
/// Element type of a tensor. Names are Rust / safetensors spelling in
/// lowercase (`"f32"`, `"bf16"`, `"bool"`). `non_exhaustive`: fp8 / int4 can
/// be added without a major version.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8, U8, I16, U16, I32, U32, I64, U64,
    F16, BF16, F32, F64,
}

impl DType {
    pub const fn size_of(self) -> usize;                        // bytes per element
    pub const fn align_of(self) -> usize;                       // == size_of for every v1 dtype
    pub const fn byte_len(self, numel: usize) -> Option<usize>; // checked; the invariant primitive
    pub const fn name(self) -> &'static str;                    // "f32"
    pub fn from_name(s: &str) -> Option<Self>;                  // case-insensitive
}
impl fmt::Display for DType { /* name() */ }
impl core::str::FromStr for DType { type Err = TensorError; }

/// Zero-copy element types. Sealed. `F16` / `BF16` gain impls behind a
/// future `tensor-half` feature. `usize` / `isize` are deliberately absent.
pub trait Element: Copy + Send + Sync + 'static + sealed::Sealed {
    const DTYPE: DType;
}
// impls: bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64

/// Arena-bound tensor header. `Copy`; 40 bytes; stored behind a reference in
/// `DataValue` so the enum stays 24 bytes.
#[derive(Clone, Copy)]
pub struct DataTensor<'a> {
    dtype: DType,
    shape: &'a [usize],
    data: &'a [u8],
}

/// Heap-owned tensor. Bytes live in an 8-byte-aligned private buffer.
#[derive(Clone)]
pub struct OwnedDataTensor {
    dtype: DType,
    shape: Box<[usize]>,
    data: AlignedBytes,        // private: Vec<u64> + byte length in v1
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ShapeOverflow,
    LengthMismatch { expected: usize, actual: usize },
    Misaligned { required: usize },
    InvalidBool,                          // Bool payload byte other than 0 / 1
    UnknownDType(String),
    UnsupportedDType(DType),              // nested conversion on F16 / BF16 without `tensor-half`
    // boundary decoder
    NotATensor,                           // not the `{"tensor": {...}}` form
    UnexpectedField(String),              // unknown key inside `tensor` (strict, see §4.6)
    InvalidBase64,
    // nested conversion
    Ragged { depth: usize },              // sibling arrays disagree on length at this depth
    Element { index: usize, expected: DType }, // leaf not representable in the dtype
}
impl fmt::Display for TensorError {}
impl std::error::Error for TensorError {}
```

### 4.2 Invariants (documented on both structs, enforced by every constructor)

1. `data.len() == dtype.byte_len(numel).unwrap()` where
   `numel = shape.iter().try_fold(1usize, checked_mul)`.
2. `data.as_ptr() as usize % dtype.align_of() == 0`.
3. If `dtype == Bool`, every byte of `data` is `0` or `1`.
4. Layout is row-major (C order), contiguous, native byte order.
5. Big-endian targets are rejected at compile time in v1.

1–3 make `as_slice::<T>()` sound; 4 makes the buffer one safetensors / ONNX /
numpy buffer.

### 4.3 API surface

| | `DataTensor<'a>` | `OwnedDataTensor` |
|---|---|---|
| wrap bytes, zero-copy | `from_bytes(dtype, &'a [usize], &'a [u8]) -> Result` | — |
| wrap typed, zero-copy | `from_slice::<T: Element>(&'a [usize], &'a [T]) -> Result` | — |
| copy bytes | `from_bytes_in(dtype, &[usize], &[u8], &'a Bump) -> Result` | `from_bytes(dtype, impl Into<Box<[usize]>>, &[u8]) -> Result` |
| copy typed | `from_slice_in::<T: Element>(&[usize], &[T], &'a Bump) -> Result` | `from_slice::<T: Element>(impl Into<Box<[usize]>>, &[T]) -> Result` |
| nested JSON → tensor | `from_nested_in(&DataValue<'_>, DType, &'a Bump) -> Result` | `from_nested(&OwnedDataValue, DType) -> Result` |
| tensor → nested JSON | `to_nested_in(&self, &'a Bump) -> Result<DataValue<'a>>` | `to_nested(&self) -> Result<OwnedDataValue>` |
| read | `dtype()`, `shape() -> &'a [usize]`, `ndim()`, `numel()`, `data() -> &'a [u8]` | same, borrowing `&self` |
| borrowed view | — | `view(&self) -> DataTensor<'_>` (zero-copy; every read-side consumer — emitter, serde, equality — is written once against `DataTensor`) |
| typed view | `as_slice::<T: Element>() -> Option<&'a [T]>` | `as_slice::<T>() -> Option<&[T]>` |
| convert | `to_owned() -> OwnedDataTensor` | `to_arena(&self, &'a Bump) -> DataTensor<'a>` |
| boundary decode | `from_json_value_in(&DataValue<'a>, &'a Bump) -> Result` | `TryFrom<&OwnedDataValue>` |
| traits | `Copy, Clone, Debug (lossy), PartialEq (bytewise)` | `Clone, Debug (lossy), PartialEq (bytewise)` |
| serde (`serde` feature) | `Serialize` | `Serialize` (no `Deserialize` in v1) |

The zero-copy typed wrap `from_slice(shape, &'a [T])` is new relative to
the first draft. It is what the element-wise path needs: an operator collects
into a `bumpalo::collections::Vec<'a, f32>`, calls `into_bump_slice()`, and
wraps it with no copy and guaranteed alignment. It also wraps ONNX Runtime
output views directly. `&[T] → &[u8]` is always a sound cast, so this
constructor cannot fail on alignment, only on length / overflow.

Both value enums, mirroring `datetime` one-for-one:

```rust
// value.rs
#[cfg(feature = "tensor")]
Tensor(&'a DataTensor<'a>),
// owned.rs
#[cfg(feature = "tensor")]
Tensor(Arc<OwnedDataTensor>),

// DataValue
pub fn is_tensor(&self) -> bool;
pub fn as_tensor(&self) -> Option<&'a DataTensor<'a>>;
pub fn tensor(t: &'a DataTensor<'a>) -> Self;                 // mirrors `datetime(dt)`
pub fn tensor_in(t: DataTensor<'a>, arena: &'a Bump) -> Self; // `Tensor(arena.alloc(t))`

// OwnedDataValue
pub fn is_tensor(&self) -> bool;
pub fn as_tensor(&self) -> Option<&OwnedDataTensor>;
pub fn tensor(t: OwnedDataTensor) -> Self;                    // wraps in Arc
impl From<OwnedDataTensor> for OwnedDataValue;               // so `owned_json!({"x": t})` works
impl From<Arc<OwnedDataTensor>> for OwnedDataValue;          // share an existing Arc
```

Opacity rules (both types): `get(..)` → `None`, `Index` → `Null`, `len()` /
`is_empty()` → `None`, `members()` / `entries()` → empty. `as_array` and
friends → `None`. No coercion anywhere.

Size guard, next to the enums:

```rust
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(core::mem::size_of::<DataValue<'static>>() == 24);
    assert!(core::mem::size_of::<OwnedDataValue>() == 32);
};
```

### 4.4 Wire format (JSON)

Compact (`Display`, `to_json_string`, `write_json_into`, `to_json_str_in`,
`serde_json::to_string`, `to_serde_value`):

```json
{"tensor":{"dtype":"f32","shape":[2,3],"data":"AACAPwAAAEAAAEBAAACAQAAAoEAAAMBA"}}
```

That is `f32 [[1,2,3],[4,5,6]]`: 24 little-endian bytes, standard base64
alphabet, `=` padding, no line breaks. The base64 alphabet contains no
JSON-special bytes, so the emitter writes it raw between quotes with no escape
scan (same trick as the datetime arm), in 3072-byte input / 4096-byte output
chunks through the existing `JsonSink`.

Pretty (`pretty()`), identical to `serde_json::to_string_pretty` of the
tagged object, which is the property the test asserts:

```json
{
  "tensor": {
    "dtype": "f32",
    "shape": [
      2,
      3
    ],
    "data": "AACAPwAAAEAAAEBAAACAQAAAoEAAAMBA"
  }
}
```

Edge cases: `shape: []` with one element; `shape: [0, 3]` with `"data": ""`.
Key order inside `tensor` is fixed (`dtype`, `shape`, `data`) so output is
byte-stable. The tag is `DataTensor::JSON_TAG`, the one definition every
producer and the decoder use.

**Why `tensor` and not `$tensor`.** The first draft used `$tensor`, citing
MongoDB Extended JSON. Checked against datalogic-rs, that was wrong twice
over. Its boundary form for foreign types is a single-key object named
after the operator with no sigil — `{"datetime": "<iso>"}`,
`{"timestamp": "<duration>"}` — and that one object is the operator call
in a rule, the form the extraction helper accepts from data, and what the
serde bridge emits, so emitter output evaluates back to the same value.
And `$` already means something else there: with the template key-escape
option on, `{"$type": ..}` emits a literal key `type`, so `{"$tensor": ..}`
in a template would silently become a plain field. `{"tensor": ..}` gets
the datetime round-trip property for free and collides with nothing. The encoder emits exactly these three keys; any future key
(e.g. `strides`) is a wire-format version change, handled by the decoder's
strictness below.

Alternatives rejected: nested lists for `data` (re-introduces the element
expansion the variant exists to avoid; cannot express `F16` / `BF16` without
a conversion); emitting `null` (silently lossy); an untagged
`{"dtype","shape","data"}` object (ambiguous with user data).

### 4.5 serde behaviour

- `Serialize` (both types): `serialize_map` with one entry `"tensor"` →
  map with `dtype` (str), `shape` (seq of u64), `data`
  (`serialize_str(base64)`, in every format — §3.7).
- `DataValueSeed` and `Deserialize for OwnedDataValue`: unchanged. The tagged
  form arrives as an `Object`.
- `serde_json_bridge`: `to_serde_value` builds the tagged
  `serde_json::Value::Object`; `from_serde_value*` never produces `Tensor`
  (parity with the parser).

### 4.6 Boundary decoder

`DataTensor::from_json_value_in` / `OwnedDataTensor::try_from(&OwnedDataValue)`:

1. `Tensor(t)` input → returned as-is (no copy on the arena side; `Arc` clone
   on the owned side).
2. Otherwise require `Object` with exactly one key, `tensor`, whose value is
   an `Object` with `dtype: String`, `shape: Array`, `data: String`. Missing
   key or wrong type → `NotATensor`. **Strict on extras:** any other key
   inside `tensor` → `UnexpectedField(name)`. Rationale: a future field such
   as `strides` changes the meaning of `data`; an old decoder must fail loudly
   rather than silently produce a wrong tensor. Key order inside `tensor`
   is not significant on input.
3. `shape` elements are read with `as_i64()` (so a whole float like `2.0` is
   accepted, matching the crate's number canonicalisation); negative or
   non-integer → `NotATensor`; product overflow → `ShapeOverflow`.
4. `DType::from_name` → `UnknownDType`; base64 decode (strict: no whitespace,
   correct padding) → `InvalidBase64`; then the normal constructor validates
   length, bool payload, and allocates aligned.

Base64 encode and decode are hand-rolled in a `pub(crate) mod base64`
(~120 lines with tests). Adding the `base64` crate was considered; the
crate's dependency policy (every dep `default-features = false`, pulled for
one thing) and the size of the encoder argue for in-tree.

### 4.7 Feature and dependency policy

```toml
[features]
tensor = []          # no dependencies; includes typed views, decoder, nested conversion
```

Additive, each optional and named for what it pulls in. `tensor-half`
shipped with v1 (a downstream byte-cell operator family asked for it); the
rest stay future work.

| Feature | Adds | Dependency | State |
|---|---|---|---|
| `tensor-half` | `Element for half::f16 / half::bf16`; nested conversion for `F16` / `BF16` | `half` | shipped |
| `tensor-ndarray` | `From<ArrayViewD<T>>`, `as_ndarray::<T>()` | `ndarray` | future |
| `tensor-safetensors` | `DataTensor::from_safetensors_view`, `OwnedDataTensor::to_safetensors` | `safetensors` | future |

### 4.7a Building a payload in place (`zeroed_bytes_in`)

`from_bytes` is zero-copy only for a buffer already aligned to
`dtype.align_of()`, and `from_bytes_in` copies. Neither serves code that
*assembles* a payload — a byte-cell operator family (stack, concat, unstack,
transpose, pad, crop, gather) moving `size_of()`-byte cells — because a
`bumpalo::collections::Vec<u8>` asks for `align_of::<u8>()`, so whether
`into_bump_slice()` satisfies a wider dtype depends on what was allocated
before it. Flaky alignment is worse than none: the operator passes its unit
test and fails in a pipeline.

**Decision.** `DataTensor::zeroed_bytes_in(dtype, shape, arena) -> Result<&mut [u8]>`
returns the zeroed, aligned payload of a tensor that does not exist yet;
the caller fills it and hands it to `from_bytes`, which wraps it with no
copy. Invariants are unaffected — the tensor is still born in a constructor,
and wrapping consumes the `&mut`, so a `Bool` payload cannot be edited
behind a live view. No element type is named, so it is also the build path
for `F16` / `BF16` without `tensor-half`. Zeroed rather than uninit: an
uninit buffer needs a public `unsafe` assume-init contract, the memset is a
linear pass rather than a copy pass, and `pad` / `crop` get their padding
from it for free.

No owned twin. The arena owns the allocation, so it can be handed out and
wrapped later; owned storage lives inside the tensor, so the equivalent
would have to be a fill closure. Add it when an owned consumer asks.

### 4.8 Nested conversion (`from_nested_in` / `to_nested_in`)

**Shape inference.** Walk depth-first. The first leaf's path fixes `ndim`
and the per-level lengths; every sibling array at depth `d` must have the
same length or the result is `Ragged { depth: d }`. A leaf at a different
depth than the first leaf is also `Ragged`. Examples:

| input | shape |
|---|---|
| `7` (bare scalar) | `[]` (0-d) |
| `[]` | `[0]` |
| `[[]]` | `[1, 0]` |
| `[[1],[2]]` | `[2, 1]` |
| `[[1,2],[3]]` | `Ragged { depth: 1 }` |
| `[1,[2]]` | `Ragged { depth: 1 }` |

Nesting depth is already bounded by the parser (`MAX_DEPTH = 256`).

**Element rules** (`Element { index, expected }` on violation; `index` is the
row-major element position):

| target dtype | accepted JSON leaf | rule |
|---|---|---|
| `Bool` | `Bool` | as-is; `Number` is **not** accepted (no truthiness) |
| `I8..I64`, `U8..U32` | `Number` | `as_i64()` must succeed (integer, or whole-valued float — the crate's existing canonicalisation) **and** be in the dtype's range; otherwise error. No truncation. |
| `U64` | `Number` | as above for values ≤ `i64::MAX`; a JSON integer above that arrives from the parser as `Float`, accepted if whole and `< 2^64` (precision above 2^53 was already lost at parse time; documented). |
| `F32`, `F64` | `Number` | `as_f64()` then `as` cast. Rounding within range is inherent and accepted (the crate itself rounds `i64 → f64` in its overflow fallbacks). If the cast result is not finite, the value did not fit and the element is refused with `Element` — no `±inf` is ever manufactured from finite input. `NaN` / `±inf` cannot occur from JSON input, so a non-finite result always means overflow. |
| `F16`, `BF16` | — | `UnsupportedDType` in the base feature; `tensor-half` lifts it |
| any | `String`, `Null`, `Object`, `DateTime`, `Duration`, `Tensor` | `Element` error. No parsing of numeric strings. |

**Out-of-range policy follows the crate's existing number rules.** Today
datavalue handles a value that does not fit in exactly two ways, and never
a third: when a fixed-width type overflows it *widens* (the parser,
`NumberValue` arithmetic, `From<u64>` and the serde visitors all fall back
from `i64` to `f64`), and when a caller asks for a narrower type than the
value fits it *refuses* (`as_i64()` returns `None` for a float that is not
whole or not in range). It never saturates, wraps, truncates, or produces
infinity from a finite input. A tensor's dtype is declared by the caller and
cannot widen, so the applicable precedent is refusal: an element that does
not fit the declared dtype is an `Element { index, expected }` error, for
integers (range or non-whole) and floats (overflow to non-finite) alike.
numpy's silent `±inf` was considered and rejected on this ground.

**Implementation.** Two passes: (1) infer shape, count elements, validate
structure; (2) allocate the exact aligned buffer (`alloc_layout` on the
arena, `AlignedBytes` on the owned side) and fill it through a per-dtype
monomorphised writer. No intermediate `Vec`.

**Reverse (`to_nested_in`).** Expands to nested `Array`s in row-major
order; 0-d → the bare scalar. Integers → `NumberValue::Integer`; `U64` above
`i64::MAX` → `Float` (the crate's existing `u64` rule); floats →
`NumberValue::from_f64` (so `2.0f32` becomes `Integer(2)`, the same
canonicalisation the parser applies); `Bool` → `Bool`. `NaN` / `±inf`
become `Float`, which the emitter renders as `null` (existing behaviour,
documented). `F16` / `BF16` → `UnsupportedDType` without `tensor-half`.
This is the one place the crate expands elements into nodes, and it is
opt-in by name.

### 4.9 Data-path coverage

The first consumer will use all four entry paths (§8). Each maps to one API:

| Path | API | Copies |
|---|---|---|
| Nested JSON arrays in the request | `from_nested_in(value, dtype, arena)` | one (unavoidable: JSON nodes → contiguous buffer) |
| Tagged base64 form in the request | `from_json_value_in(value, arena)` | one (base64 decode into aligned arena memory) |
| Binary buffers (safetensors, mmap, runtime outputs) | `from_bytes(dtype, shape, &'a [u8])` / `from_slice(shape, &'a [T])` | zero |
| Built element-wise by operators | `BumpVec<'a, T>` → `into_bump_slice()` → `from_slice(shape, slice)` | zero |
| Back to the client as JSON | `Display` / `Serialize` (tagged base64), or `to_nested_in` when the client wants plain arrays | one |
| Into ONNX Runtime | `as_slice::<T>()` → `&[T]` | zero |

---

## 5. Implementation plan

Estimated size: ~1300–1600 lines including tests. One PR with the commits
below, or two PRs split after step 3 if review prefers.

### Step 1 — Core types (`src/tensor.rs`, `src/base64.rs`)

- `DType` with `size_of / align_of / byte_len / name / from_name / Display /
  FromStr`.
- `TensorError` with `Display` + `Error`.
- `Element` (sealed) and the single `unsafe` cast helper, with the soundness
  comment referencing invariants 1–3.
- `DataTensor<'a>`: `from_bytes`, `from_slice`, `from_bytes_in` (via
  `Bump::alloc_layout`), `from_slice_in`, accessors, `as_slice`, `Debug`,
  `PartialEq`.
- `OwnedDataTensor` with `AlignedBytes`: `from_bytes`, `from_slice`,
  accessors, `as_slice`, `Debug`, `PartialEq`, `to_arena`;
  `DataTensor::to_owned`.
- `base64::encode_into<S: JsonSink>` (chunked) and `base64::decode(&str)`.
- `compile_error!` for big-endian.
- Tests: RFC 4648 vectors + round trip for every length 0..=64; every dtype's
  `size_of` / `name` round trip; construction rejects length mismatch,
  overflow, misalignment (deliberately offset input), bad bool bytes; 0-d and
  zero-size shapes; typed view `None` on mismatch and correct otherwise;
  `Debug` output is bounded; `from_slice` on a `BumpVec::into_bump_slice()`
  result is zero-copy (pointer equality).

### Step 2 — Enum integration (`value.rs`, `owned.rs`, `lib.rs`, `Cargo.toml`)

- Variants, `is_tensor / as_tensor / tensor / tensor_in`, `From` impls,
  `PartialEq` arms, `to_owned` / `to_arena` arms (three arena bumps: shape,
  aligned data, header).
- Size `const` assertions.
- Re-exports: `DataTensor, OwnedDataTensor, DType, Element, TensorError`.
- Tests: arena → owned → arena preserves dtype/shape/bytes and equality;
  opacity of `get / Index / len / members / entries`; `Copy` of
  `DataValue::Tensor`; `owned_json!({"t": tensor})`; `Arc` clone shares the
  buffer (pointer equality).

### Step 3 — Rendering (`emit.rs`)

- `write_tensor` and `write_tensor_pretty`, each called from both the
  `DataValue` and `OwnedDataValue` arms (four arms, two helpers), following
  `write_datetime`.
- Tests: exact compact string for the §4.4 example; `pretty()` equals
  `serde_json::to_string_pretty(&v.to_serde_value())`; empty and 0-d cases;
  `to_json_str_in` produces the same bytes as `Display`.

### Step 4 — serde (`ser.rs`, `serde_json_bridge.rs`)

- `Serialize` for both tensor types, `data` base64 in every format; enum
  arms delegate.
- `to_serde_value` arms produce the tagged object.
- Tests: `serde_json::to_string(&v) == v.to_string()`; a minimal in-test
  serializer with `is_human_readable() == false` observes the same base64 a
  JSON one does; `from_serde_value` of the tagged object stays `Object`.

### Step 5 — Boundary decoder

- `DataTensor::from_json_value_in`, `TryFrom<&OwnedDataValue> for
  OwnedDataTensor`.
- Tests: `decode(encode(t)) == t` for every dtype; malformed inputs map to
  the right `TensorError` (including an extra key → `UnexpectedField`); a
  parsed `{"tensor": …}` is still an `Object` until the helper is called;
  adversarial shape overflows are rejected before any allocation; `2.0` in
  `shape` is accepted, `-1` and `1.5` are not.

### Step 6 — Nested conversion

- `from_nested_in` / `from_nested`, `to_nested_in` / `to_nested`, per §4.8.
- Tests: every table row in §4.8 (shapes and element rules); round trip
  nested → tensor → nested for every native dtype; `u64` above `i64::MAX`
  both directions; `Bool` from `Number` rejected; numeric string rejected;
  `F16` → `UnsupportedDType`; `index` in `Element` errors is the row-major
  position; deep nesting near `MAX_DEPTH`; `f32` overflow: `1e39` into `F32`
  is an `Element` error, `f32::MAX` as an `f64` round-trips, and a value that
  rounds down to `f32::MAX` stays finite and is accepted; `i32` from `2^31`
  is an `Element` error, from `2^31 - 1` succeeds.

### Step 7 — Docs, CI, release

- README: feature table row; "Tensor" paragraph under Design Notes (wire
  format, opacity, the boundary decoder, the typed-decode rule from §3.15).
- `CLAUDE.md`: variant table row; `Tensor` is the one `Arc`-shared owned
  payload and the one `&'a`-indirected arena payload; the §3.14 note about a
  future extension slot; the §3.15 rule so future PRs know where the
  coercion line is.
- CI: add `cargo clippy --features tensor` + `cargo test --features tensor`
  so `tensor` without `serde` is compiled (today only default and
  `--all-features` run).
- `cargo bench --bench compare --features serde_json -- --quick` before and
  after on twitter / citm / canada to confirm the extra match arm is noise.
- Bump to `0.3.0`.

### Acceptance (from the issue, extended)

- `cargo test`, `cargo test --all-features`, `cargo test --features tensor`;
  `clippy -D warnings` on all three; `fmt --check`.
- `DataValue` remains `Copy`; `const` assertions hold (24 / 32 bytes).
- `as_slice::<T>()` on a tensor built by any constructor never returns a
  misaligned slice (tested with a deliberately offset input).
- Parser and `Deserialize` produce no `Tensor` for any input.
- Every §4.8 table row has a test.

---

## 6. Consumer follow-ups (`datalogic-rs`, separate issue)

Every arm that names `Duration` today needs a `Tensor` arm under
`#[cfg(feature = "tensor")]`. From the current tree:

| File | Today (datetime) | Suggested tensor semantics |
|---|---|---|
| `arena/value/mod.rs` truthiness | `true` | `numel() > 0` (empty array is falsy in JSONLogic; a zero-size tensor should be too) |
| `operators/truthy.rs` (owned) | `true` | same |
| `arena/value/coercion.rs` | `None` | `None` (no coercion) |
| `operators/inspect.rs` type name | `"datetime"` / `"duration"` | `"tensor"` |
| `operators/array/sort.rs` rank | 3 | own rank after `object`; bytewise within rank |
| `compile/optimize/cse.rs` hash / eq | `format!("{d:?}")` | hash `dtype`, `shape`, `data()`; eq via `PartialEq` (do **not** hash the lossy `Debug`) |
| `serde_bridge.rs` | datetime sentinel | `to_serde_value()` tagged form (provided by datavalue) |
| `engine/mod.rs` owned → arena | copy | `to_arena` |
| `bindings/python/src/conv.rs` | ISO string | tagged dict in v1; later zero-copy numpy via `as_slice` + `numpy` crate |
| `node/prelit.rs`, `node/populate.rs`, `operators/error_handling.rs`, `operators/datetime/mod.rs`, `arena/value/conversion.rs` | various | need arms; not analysed in detail |

The operator family (make / reshape / read back) becomes thin: `make` is
`from_nested_in` with a dtype argument, `read back` is `to_nested_in`,
`reshape` is a new header over the same bytes (`from_bytes` with a new
shape, zero-copy). Each should be a few dozen lines.

---

## 7. Explicitly deferred

- Arithmetic, element indexing, broadcasting, strides / views (private
  fields keep strides possible later without a break).
- dtype-to-dtype conversions, including `F16` / `BF16` → `F32`.
- dtype inference from JSON (the caller always declares it).
- fp8 (`E4M3`, `E5M2`), int4: additive later via `non_exhaustive` +
  `byte_len`.
- `Deserialize` recognising `tensor` (deliberately, for parser parity).
- Big-endian support (compile error for now).
- A single-allocation header + shape + data layout on the arena side
  (micro-optimisation; three bumps per tensor is fine).
- `Hash` / `Ord` (the value enums have neither).
- A `Bytes` variant for untyped binary payloads (a tensor is typed bytes; a
  plain blob would be a separate proposal and a candidate for the §3.14
  extension slot discussion).

---

## 8. Decisions taken (2026-09-12)

| Decision | Outcome |
|---|---|
| Variant strategy | Concrete `Tensor` variant. Extension slot deferred; trigger recorded in `CLAUDE.md`. |
| Owned payload | `Arc<OwnedDataTensor>`. |
| v1 scope beyond the issue | Typed views, boundary decoder, and nested conversion all in v1. (The answer also ticked "bytes-only"; read as "all three additions", consistent with the data-path answer. Flag if that reading is wrong.) |
| First consumer's data paths | All four: nested JSON arrays, tagged base64, binary buffers, element-wise construction. §4.9 maps each to an API. |
| Big-endian | `compile_error!` in v1. |
| Naming | `DataTensor<'a>` / `OwnedDataTensor`. |
| Version | `0.3.0`. |
| Decoder strictness | Unknown keys inside `tensor` are an error. |
| Wire tag | `tensor`, not `$tensor`: matches datalogic's `{"datetime": ..}` sentinel and avoids its `$` key-escape. Exposed as `DataTensor::JSON_TAG`. |
| Out-of-range elements in nested conversion | Refuse with `Element` error, following the crate's `as_i64()` precedent (no saturation, no wrap, no manufactured `±inf`). |
