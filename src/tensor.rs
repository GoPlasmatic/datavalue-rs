//! `DataTensor` / `OwnedDataTensor` — opaque n-dimensional typed buffers
//! behind the `tensor` feature.
//!
//! A tensor is a [`DType`], a shape, and a row-major contiguous byte buffer
//! in native byte order. The JSON parser never produces one (JSON has no
//! tensor); consumers build them at an operator boundary from typed slices,
//! from nested JSON arrays ([`DataTensor::from_nested_in`]), or from the
//! tagged wire form ([`DataTensor::from_json_value_in`]). Rendering emits
//! `{"tensor":{"dtype":"f32","shape":[2,3],"data":"<base64>"}}`.
//!
//! Invariants, enforced by every constructor and relied on by
//! [`DataTensor::as_slice`]:
//!
//! 1. `data.len() == dtype.byte_len(numel)` where `numel` is the checked
//!    product of `shape`.
//! 2. `data` is aligned to `dtype.align_of()`.
//! 3. A `Bool` payload holds only `0` / `1` bytes.
//!
//! Separately from those, [`MAX_RANK`] bounds `shape.len()`. It is not a
//! soundness invariant but a resource limit: it keeps the shape-driven
//! recursions inside the stack budget the parser's own depth cap gives
//! nested JSON, so a tagged payload from the wire cannot be expanded into
//! a stack overflow.
//!
//! `F16` / `BF16` are carried as bytes by the base feature and gain typed
//! views and nested-array conversion under `tensor-half`, which adds the
//! `half` dependency and nothing else; [`DType::has_native_element`] is the
//! gate to branch on.
//!
//! The crate carries bytes and typed *views* of them. It does not convert
//! between dtypes and does no arithmetic; those belong to consumers. The
//! one place elements are decoded is the nested-array conversion, which is
//! a typed decode into a caller-declared dtype with an explicit error for
//! anything that does not fit — never a coercion.

use core::alloc::Layout;
use core::fmt;
use core::mem::{align_of, size_of};

use bumpalo::Bump;

use crate::base64;
use crate::number::NumberValue;
use crate::owned::OwnedDataValue;
use crate::value::DataValue;

// ---- DType -----------------------------------------------------------------

/// Element type of a tensor. Names use the Rust / safetensors spelling in
/// lowercase (`"f32"`, `"bf16"`, `"bool"`). `non_exhaustive`: fp8 and int4
/// can be added without a major version.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    /// Every dtype, in declaration order.
    pub const ALL: [DType; 13] = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];

    /// Bytes per element.
    #[inline]
    pub const fn size_of(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 => 8,
        }
    }

    /// Required alignment of the data buffer. Equal to [`size_of`] for
    /// every current dtype.
    ///
    /// [`size_of`]: DType::size_of
    #[inline]
    pub const fn align_of(self) -> usize {
        self.size_of()
    }

    /// Byte length of a buffer holding `numel` elements, or `None` on
    /// overflow. This is the primitive the length invariant is defined in
    /// terms of, so packed sub-byte dtypes can be added later without
    /// redefining it.
    #[inline]
    pub const fn byte_len(self, numel: usize) -> Option<usize> {
        numel.checked_mul(self.size_of())
    }

    /// Whether an [`Element`] impl exists for this dtype, i.e. whether typed
    /// views and nested-array conversion are available. This is the gate to
    /// branch on: `F16` / `BF16` report `false` until the `tensor-half`
    /// feature is enabled, and `true` once it is, with no other change to
    /// how a tensor of that dtype behaves. Carrying the bytes — building,
    /// rendering, decoding the wire form — never depends on it.
    #[inline]
    pub const fn has_native_element(self) -> bool {
        if cfg!(feature = "tensor-half") {
            true
        } else {
            !matches!(self, DType::F16 | DType::BF16)
        }
    }

    /// Wire name: `"bool"`, `"i8"`, …, `"bf16"`, `"f64"`.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::I16 => "i16",
            DType::U16 => "u16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::U64 => "u64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
        }
    }

    /// Inverse of [`name`](DType::name); ASCII case-insensitive so
    /// safetensors-style `"F32"` / `"BF16"` are accepted too.
    pub fn from_name(s: &str) -> Option<Self> {
        DType::ALL
            .iter()
            .copied()
            .find(|d| d.name().eq_ignore_ascii_case(s))
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl core::str::FromStr for DType {
    type Err = TensorError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| TensorError::UnknownDType(s.to_string()))
    }
}

// ---- Errors ----------------------------------------------------------------

/// Why a tensor could not be built.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// The product of the shape (or its byte length) overflows `usize`.
    ShapeOverflow,
    /// `data.len()` disagrees with `shape` × `dtype`.
    LengthMismatch { expected: usize, actual: usize },
    /// A caller-supplied buffer is not aligned to `dtype.align_of()`.
    Misaligned { required: usize },
    /// A `Bool` payload contains a byte other than `0` or `1`.
    InvalidBool,
    /// `dtype` name not recognised.
    UnknownDType(String),
    /// Nested conversion asked for a dtype the base feature cannot decode
    /// (`F16` / `BF16` without `tensor-half`).
    UnsupportedDType(DType),
    /// Boundary decoder: the value is not the `{"tensor": {...}}` form.
    NotATensor,
    /// Boundary decoder: an unknown key inside `tensor`. Strict on purpose —
    /// a future field such as `strides` would change the meaning of `data`.
    UnexpectedField(String),
    /// Boundary decoder: `data` is not valid base64.
    InvalidBase64,
    /// Nested conversion: sibling arrays disagree on length, or a leaf and
    /// an array occur at the same depth.
    Ragged { depth: usize },
    /// Nested conversion: the leaf at row-major position `index` is not
    /// representable in `expected` (wrong JSON type, non-whole float for an
    /// integer dtype, out of range, or overflow to a non-finite float).
    Element { index: usize, expected: DType },
    /// `shape.len()` exceeds [`MAX_RANK`].
    RankTooHigh { max: usize, actual: usize },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeOverflow => f.write_str("tensor shape overflows usize"),
            TensorError::LengthMismatch { expected, actual } => write!(
                f,
                "tensor data length {actual} does not match shape ({expected} bytes expected)"
            ),
            TensorError::Misaligned { required } => {
                write!(f, "tensor data is not aligned to {required} bytes")
            }
            TensorError::InvalidBool => {
                f.write_str("bool tensor payload contains a byte other than 0 or 1")
            }
            TensorError::UnknownDType(s) => write!(f, "unknown dtype {s:?}"),
            TensorError::UnsupportedDType(d) => {
                write!(f, "dtype {d} is not supported by this conversion")
            }
            TensorError::NotATensor => f.write_str("value is not a {\"tensor\": {...}} object"),
            TensorError::UnexpectedField(k) => write!(f, "unexpected field {k:?} in tensor"),
            TensorError::InvalidBase64 => f.write_str("tensor data is not valid base64"),
            TensorError::Ragged { depth } => {
                write!(f, "nested arrays are ragged at depth {depth}")
            }
            TensorError::Element { index, expected } => {
                write!(f, "element {index} is not representable as {expected}")
            }
            TensorError::RankTooHigh { max, actual } => {
                write!(f, "tensor rank {actual} exceeds the maximum of {max}")
            }
        }
    }
}

impl std::error::Error for TensorError {}

// ---- Element ---------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
}

/// Scalar types that can be viewed zero-copy from a tensor's bytes. Sealed;
/// `half::f16` / `half::bf16` are included with the `tensor-half` feature.
/// `usize` / `isize` are deliberately absent (platform-dependent width).
pub trait Element: Copy + Send + Sync + 'static + sealed::Sealed {
    const DTYPE: DType;
}

macro_rules! element {
    ($($t:ty => $d:ident),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl Element for $t {
            const DTYPE: DType = DType::$d;
        }
    )*};
}
element!(
    bool => Bool,
    i8 => I8, u8 => U8, i16 => I16, u16 => U16,
    i32 => I32, u32 => U32, i64 => I64, u64 => U64,
    f32 => F32, f64 => F64,
);

// Both are `#[repr(transparent)]` over `u16`, so every bit pattern is a
// valid value (NaN payloads included) and the size / alignment the dtype
// table declares is the type's own — the same argument the other elements
// rest on.
#[cfg(feature = "tensor-half")]
element!(half::f16 => F16, half::bf16 => BF16);

/// View a typed slice as bytes. Every `Element` is a plain scalar with no
/// padding, so every byte is initialised.
#[inline]
fn bytes_of<T: Element>(s: &[T]) -> &[u8] {
    // SAFETY: see above; the byte length is the slice's own size in memory.
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u8>(), core::mem::size_of_val(s)) }
}

/// The one reinterpretation in the crate. `data` must satisfy invariants
/// 1–3 for `dtype`.
#[inline]
fn cast_slice<T: Element>(dtype: DType, data: &[u8]) -> Option<&[T]> {
    if dtype != T::DTYPE {
        return None;
    }
    if data.is_empty() {
        return Some(&[]);
    }
    debug_assert_eq!(data.len() % size_of::<T>(), 0);
    debug_assert_eq!(data.as_ptr() as usize % align_of::<T>(), 0);
    // SAFETY: invariant 1 gives a whole number of `T`s, invariant 2 gives
    // alignment, and every bit pattern is a valid integer or float; for
    // `bool`, invariant 3 restricts the bytes to 0 / 1.
    Some(unsafe {
        core::slice::from_raw_parts(data.as_ptr().cast::<T>(), data.len() / size_of::<T>())
    })
}

/// Mutable counterpart, used only by the fill paths on zero-initialised
/// memory (all-zero bytes are a valid value of every `Element`).
#[inline]
fn cast_slice_mut<T: Element>(data: &mut [u8]) -> &mut [T] {
    if data.is_empty() {
        return &mut [];
    }
    debug_assert_eq!(data.len() % size_of::<T>(), 0);
    debug_assert_eq!(data.as_ptr() as usize % align_of::<T>(), 0);
    // SAFETY: as for `cast_slice`; the caller guarantees alignment and a
    // whole number of elements, and the memory is initialised (zeroed).
    unsafe {
        core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<T>(), data.len() / size_of::<T>())
    }
}

// ---- Invariant checks ----------------------------------------------------------

/// Largest `shape.len()` any tensor may have. The shape drives two
/// recursions — `fill_rec` when decoding nested arrays and `emit_rec` when
/// expanding back into them — so an unbounded rank is an unbounded stack
/// depth. Matches the parser's `MAX_DEPTH`, and is orders of magnitude past
/// any real tensor.
pub const MAX_RANK: usize = 256;

/// Bytes a tensor of this dtype and shape must hold; `ShapeOverflow` if the
/// element count or the byte count overflows `usize`, `RankTooHigh` if the
/// shape has more than [`MAX_RANK`] dimensions.
///
/// Every constructor reaches this function — directly, through `check_len`,
/// or through `nested_prepare` / `tagged_prepare` — and reaches it before
/// allocating and before any shape-driven recursion, so the rank cap is
/// checked exactly once per tensor, on every path.
fn expected_byte_len(dtype: DType, shape: &[usize]) -> Result<usize, TensorError> {
    if shape.len() > MAX_RANK {
        return Err(TensorError::RankTooHigh {
            max: MAX_RANK,
            actual: shape.len(),
        });
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .and_then(|numel| dtype.byte_len(numel))
        .ok_or(TensorError::ShapeOverflow)
}

/// Invariant 1.
fn check_len(dtype: DType, shape: &[usize], data: &[u8]) -> Result<(), TensorError> {
    let expected = expected_byte_len(dtype, shape)?;
    if data.len() != expected {
        return Err(TensorError::LengthMismatch {
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

/// Invariant 2, for caller-supplied buffers. Buffers the crate allocates are
/// aligned by construction and skip this.
fn check_align(dtype: DType, data: &[u8]) -> Result<(), TensorError> {
    if !data.is_empty() && !(data.as_ptr() as usize).is_multiple_of(dtype.align_of()) {
        return Err(TensorError::Misaligned {
            required: dtype.align_of(),
        });
    }
    Ok(())
}

/// Invariant 3. The one home for per-dtype payload rules (a packed or
/// reserved-pattern dtype would add its rule here). Typed-slice constructors
/// skip it: the element type already guarantees it.
fn validate_payload(dtype: DType, data: &[u8]) -> Result<(), TensorError> {
    if dtype == DType::Bool && !data.iter().all(|&b| b <= 1) {
        return Err(TensorError::InvalidBool);
    }
    Ok(())
}

// ---- Allocation -------------------------------------------------------------------

/// Zero-filled arena bytes aligned to `align`. Zero is a valid value of
/// every `Element`, so the result can be viewed as `&mut [T]` immediately.
#[allow(clippy::mut_from_ref)]
fn alloc_zeroed_aligned(arena: &Bump, len: usize, align: usize) -> &mut [u8] {
    if len == 0 {
        return &mut [];
    }
    let layout = Layout::from_size_align(len, align).expect("tensor byte length fits in isize");
    let ptr = arena.alloc_layout(layout);
    // SAFETY: `ptr` is a fresh, exclusively owned allocation of `len` bytes;
    // every byte is initialised before the slice is formed.
    unsafe {
        core::ptr::write_bytes(ptr.as_ptr(), 0, len);
        core::slice::from_raw_parts_mut(ptr.as_ptr(), len)
    }
}

/// Copy `data` into the arena at alignment `align`.
fn alloc_aligned_copy<'a>(arena: &'a Bump, data: &[u8], align: usize) -> &'a [u8] {
    if data.is_empty() {
        return &[];
    }
    let layout =
        Layout::from_size_align(data.len(), align).expect("tensor byte length fits in isize");
    let ptr = arena.alloc_layout(layout);
    // SAFETY: fresh allocation of `data.len()` bytes; source and destination
    // cannot overlap.
    unsafe {
        core::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
        core::slice::from_raw_parts(ptr.as_ptr(), data.len())
    }
}

/// Alignment of every owned buffer. Pinned against the dtype table so a
/// wider dtype cannot silently break the owned side's soundness argument.
const MAX_ALIGN: usize = 8;

const _: () = {
    assert!(MAX_ALIGN == align_of::<u64>());
    let mut i = 0;
    while i < DType::ALL.len() {
        assert!(DType::ALL[i].align_of() <= MAX_ALIGN);
        i += 1;
    }
};

/// `MAX_ALIGN`-aligned byte storage for the owned tensor. `Vec<u8>` carries
/// no alignment guarantee, so the bytes live in `u64` words; private, so the
/// representation can change without a break.
#[derive(Clone)]
struct AlignedBytes {
    words: Vec<u64>,
    len: usize,
}

impl AlignedBytes {
    fn zeroed(len: usize) -> Self {
        AlignedBytes {
            words: vec![0u64; len.div_ceil(MAX_ALIGN)],
            len,
        }
    }

    fn copy_from(bytes: &[u8]) -> Self {
        let mut out = Self::zeroed(bytes.len());
        out.as_mut_bytes().copy_from_slice(bytes);
        out
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `words` holds at least `len` initialised bytes and `u8`
        // has no alignment requirement.
        unsafe { core::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }

    #[inline]
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as for `as_bytes`, with exclusive access.
        unsafe { core::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), self.len) }
    }
}

// ---- DataTensor ------------------------------------------------------------

/// Arena-bound tensor: dtype, shape, and a row-major contiguous byte
/// buffer. `Copy`; 40 bytes; [`DataValue::Tensor`] holds it behind a
/// reference so the value enum stays 24 bytes. It is also the borrowed view
/// of an [`OwnedDataTensor`] (see [`OwnedDataTensor::view`]), so every
/// read-side consumer is written once, against this type.
///
/// Fields are private so the invariants in the module docs always hold.
#[derive(Clone, Copy)]
pub struct DataTensor<'a> {
    dtype: DType,
    shape: &'a [usize],
    data: &'a [u8],
}

/// Inner keys of the wire form, shared by the serde producer and the decoder.
pub(crate) const KEY_DTYPE: &str = "dtype";
pub(crate) const KEY_SHAPE: &str = "shape";
pub(crate) const KEY_DATA: &str = "data";

impl<'a> DataTensor<'a> {
    /// Key of the single-entry object that carries a tensor through JSON:
    /// `{"tensor": {"dtype": .., "shape": [..], "data": ..}}`. Named after
    /// the operator, with no sigil, following datalogic's `{"datetime": ..}`
    /// / `{"timestamp": ..}` sentinels — a `$` prefix would collide with
    /// datalogic's template key-escape (`{"$type": ..}` emits a literal `type`).
    pub const JSON_TAG: &'static str = "tensor";

    #[inline]
    fn new_unchecked(dtype: DType, shape: &'a [usize], data: &'a [u8]) -> Self {
        DataTensor { dtype, shape, data }
    }

    /// Wrap caller-owned bytes with no copy (an mmap'd safetensors entry, a
    /// runtime output, a slice inside an `Arc`). Validates length,
    /// alignment, and `Bool` payload.
    ///
    /// Zero-copy holds only for a buffer already aligned to
    /// `dtype.align_of()`; anything else is `Misaligned`, because that
    /// alignment is what makes [`as_slice`] sound. To assemble a payload
    /// yourself and still avoid a copy, allocate it with
    /// [`zeroed_bytes_in`] and wrap the result here;
    /// [`from_bytes_in`] is the copying fallback for a buffer you do not
    /// control.
    ///
    /// [`as_slice`]: DataTensor::as_slice
    /// [`zeroed_bytes_in`]: DataTensor::zeroed_bytes_in
    /// [`from_bytes_in`]: DataTensor::from_bytes_in
    pub fn from_bytes(
        dtype: DType,
        shape: &'a [usize],
        data: &'a [u8],
    ) -> Result<Self, TensorError> {
        check_len(dtype, shape, data)?;
        check_align(dtype, data)?;
        validate_payload(dtype, data)?;
        Ok(Self::new_unchecked(dtype, shape, data))
    }

    /// Wrap a typed slice with no copy. Alignment and payload are guaranteed
    /// by the element type, so only the length can fail. This is the
    /// zero-copy path for `bumpalo::collections::Vec<T>::into_bump_slice()`.
    pub fn from_slice<T: Element>(shape: &'a [usize], data: &'a [T]) -> Result<Self, TensorError> {
        let bytes = bytes_of(data);
        check_len(T::DTYPE, shape, bytes)?;
        Ok(Self::new_unchecked(T::DTYPE, shape, bytes))
    }

    /// Copy `shape` and `data` into `arena`. The copy is aligned for
    /// `dtype`, so misaligned input (a wire buffer) becomes a valid tensor.
    /// Copies to guarantee alignment. When the payload is yours to build,
    /// [`zeroed_bytes_in`] + [`from_bytes`] does it in one pass instead.
    ///
    /// [`zeroed_bytes_in`]: DataTensor::zeroed_bytes_in
    /// [`from_bytes`]: DataTensor::from_bytes
    pub fn from_bytes_in(
        dtype: DType,
        shape: &[usize],
        data: &[u8],
        arena: &'a Bump,
    ) -> Result<Self, TensorError> {
        check_len(dtype, shape, data)?;
        validate_payload(dtype, data)?;
        let shape: &'a [usize] = arena.alloc_slice_copy(shape);
        let data = alloc_aligned_copy(arena, data, dtype.align_of());
        Ok(Self::new_unchecked(dtype, shape, data))
    }

    /// Zeroed arena bytes, aligned for `dtype` and sized for `dtype` ×
    /// `shape`: the payload of a tensor that does not exist yet. Fill it in
    /// place and hand it to [`from_bytes`], which wraps it with no copy.
    ///
    /// This is the one route that builds a payload in a single pass. A
    /// `bumpalo::collections::Vec<u8>` cannot stand in: it asks for
    /// `align_of::<u8>()`, so whether `into_bump_slice()` happens to satisfy
    /// a wider dtype depends on what was allocated before it, and
    /// [`from_bytes`] rejects it as `Misaligned` when it does not.
    ///
    /// The tensor still comes into existence through a constructor, so all
    /// three invariants are checked before anything can view the bytes —
    /// handing out `&mut [u8]` cannot break a `Bool` payload, because
    /// wrapping it consumes the borrow. No element type is named, so this is
    /// also the build path for `F16` / `BF16` without `tensor-half`.
    ///
    /// ```
    /// # use bumpalo::Bump;
    /// # use datavalue_rs::{DType, DataTensor};
    /// let arena = Bump::new();
    /// let shape: &[usize] = arena.alloc_slice_copy(&[2, 2]);
    /// let buf = DataTensor::zeroed_bytes_in(DType::F32, shape, &arena)?;
    /// buf[..4].copy_from_slice(&1.5f32.to_ne_bytes());
    /// let t = DataTensor::from_bytes(DType::F32, shape, buf)?;
    /// assert_eq!(t.as_slice::<f32>().unwrap(), &[1.5, 0.0, 0.0, 0.0]);
    /// # Ok::<(), datavalue_rs::TensorError>(())
    /// ```
    ///
    /// [`from_bytes`]: DataTensor::from_bytes
    // Sound for the same reason bumpalo's own `alloc_slice_fill_*` are: the
    // memory is a fresh, exclusively owned allocation.
    #[allow(clippy::mut_from_ref)]
    pub fn zeroed_bytes_in(
        dtype: DType,
        shape: &[usize],
        arena: &'a Bump,
    ) -> Result<&'a mut [u8], TensorError> {
        // Same chokepoint as every other constructor: rank and overflow are
        // rejected here, before a byte is allocated.
        let len = expected_byte_len(dtype, shape)?;
        Ok(alloc_zeroed_aligned(arena, len, dtype.align_of()))
    }

    /// Copy a typed slice into `arena`.
    pub fn from_slice_in<T: Element>(
        shape: &[usize],
        data: &[T],
        arena: &'a Bump,
    ) -> Result<Self, TensorError> {
        check_len(T::DTYPE, shape, bytes_of(data))?;
        let shape: &'a [usize] = arena.alloc_slice_copy(shape);
        let data: &'a [T] = arena.alloc_slice_copy(data);
        Ok(Self::new_unchecked(T::DTYPE, shape, bytes_of(data)))
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements (product of the shape; `1` for a 0-d tensor).
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Raw bytes: native byte order, row-major, aligned to
    /// `dtype().align_of()`.
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Zero-copy typed view. `None` if `T` is not the tensor's dtype.
    #[inline]
    pub fn as_slice<T: Element>(&self) -> Option<&'a [T]> {
        cast_slice(self.dtype, self.data)
    }

    /// Deep-copy into an [`OwnedDataTensor`] that outlives the arena.
    pub fn to_owned(&self) -> OwnedDataTensor {
        OwnedDataTensor {
            dtype: self.dtype,
            shape: self.shape.into(),
            data: AlignedBytes::copy_from(self.data),
        }
    }

    /// Build a tensor from nested JSON arrays (or a bare scalar for a 0-d
    /// tensor), decoding each leaf into `dtype`. See the module docs for the
    /// element rules; the shape is inferred from the nesting.
    pub fn from_nested_in(
        v: &DataValue<'_>,
        dtype: DType,
        arena: &'a Bump,
    ) -> Result<Self, TensorError> {
        let (shape, len) = nested_prepare(v, dtype)?;
        let data = alloc_zeroed_aligned(arena, len, dtype.align_of());
        fill_tensor(v, dtype, &shape, data)?;
        let shape: &'a [usize] = arena.alloc_slice_copy(&shape);
        Ok(Self::new_unchecked(dtype, shape, data))
    }

    /// Expand into nested `Array`s of `Number` / `Bool` nodes in `arena`
    /// (a bare scalar for a 0-d tensor). This is the one place the crate
    /// expands elements into nodes, and it is opt-in by name.
    ///
    /// Every element is copied, so the result borrows `arena` only — it may
    /// outlive the tensor it came from.
    pub fn to_nested_in<'b>(&self, arena: &'b Bump) -> Result<DataValue<'b>, TensorError> {
        emit_nested(*self, &ArenaSink(arena))
    }

    /// Recognise the `{"tensor": {...}}` wire form (see [`DataTensor::JSON_TAG`])
    /// on an already-parsed tree and decode it into `arena`. The parser never
    /// calls this; it is the `DataDateTime::parse` of tensors. A
    /// `DataValue::Tensor` input is returned as-is with no copy.
    pub fn from_json_value_in(v: &DataValue<'a>, arena: &'a Bump) -> Result<Self, TensorError> {
        if let DataValue::Tensor(t) = v {
            return Ok(**t);
        }
        let tagged = tagged_prepare(v)?;
        let shape: &'a [usize] = arena.alloc_slice_copy(&tagged.shape);
        let data = alloc_zeroed_aligned(arena, tagged.byte_len, tagged.dtype.align_of());
        decode_tagged_into(tagged.dtype, tagged.data, data)?;
        Ok(Self::new_unchecked(tagged.dtype, shape, data))
    }
}

impl fmt::Debug for DataTensor<'_> {
    /// Deliberately lossy: dtype, shape, and byte count, never the payload.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataTensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("bytes", &self.data.len())
            .finish()
    }
}

impl PartialEq for DataTensor<'_> {
    /// Structural: dtype, shape, and bytes. Note this is bytewise on the
    /// payload, so identical `NaN` patterns compare equal and `0.0` /
    /// `-0.0` do not — unlike `NumberValue`.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.shape == other.shape && self.data == other.data
    }
}

impl Eq for DataTensor<'_> {}

// ---- OwnedDataTensor -------------------------------------------------------

/// Heap-owned tensor. Mirrors [`DataTensor`] one-for-one; the byte buffer
/// is 8-byte aligned by construction, so typed views are always available.
/// [`OwnedDataValue::Tensor`] holds it in an `Arc` — the buffer is
/// immutable and potentially large, so clones of the value tree share it.
#[derive(Clone)]
pub struct OwnedDataTensor {
    dtype: DType,
    shape: Box<[usize]>,
    data: AlignedBytes,
}

impl OwnedDataTensor {
    /// Copy `data` into aligned owned storage. Validates length and `Bool`
    /// payload; the input need not be aligned.
    pub fn from_bytes(
        dtype: DType,
        shape: impl Into<Box<[usize]>>,
        data: &[u8],
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        check_len(dtype, &shape, data)?;
        validate_payload(dtype, data)?;
        Ok(OwnedDataTensor {
            dtype,
            shape,
            data: AlignedBytes::copy_from(data),
        })
    }

    /// Copy a typed slice into aligned owned storage.
    pub fn from_slice<T: Element>(
        shape: impl Into<Box<[usize]>>,
        data: &[T],
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let bytes = bytes_of(data);
        check_len(T::DTYPE, &shape, bytes)?;
        Ok(OwnedDataTensor {
            dtype: T::DTYPE,
            shape,
            data: AlignedBytes::copy_from(bytes),
        })
    }

    /// Zero-copy borrowed view. Owned storage already satisfies every
    /// invariant, so this is the read-side type for both forms.
    #[inline]
    pub fn view(&self) -> DataTensor<'_> {
        DataTensor::new_unchecked(self.dtype, &self.shape, self.data.as_bytes())
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements (product of the shape; `1` for a 0-d tensor).
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Raw bytes: native byte order, row-major, 8-byte aligned.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// Zero-copy typed view. `None` if `T` is not the tensor's dtype.
    #[inline]
    pub fn as_slice<T: Element>(&self) -> Option<&[T]> {
        cast_slice(self.dtype, self.data.as_bytes())
    }

    /// Copy into `arena` and return the borrowed form.
    pub fn to_arena<'a>(&self, arena: &'a Bump) -> DataTensor<'a> {
        let v = self.view();
        let shape: &'a [usize] = arena.alloc_slice_copy(v.shape);
        let data = alloc_aligned_copy(arena, v.data, v.dtype.align_of());
        DataTensor::new_unchecked(v.dtype, shape, data)
    }

    /// Owned twin of [`DataTensor::from_nested_in`].
    pub fn from_nested(v: &OwnedDataValue, dtype: DType) -> Result<Self, TensorError> {
        let (shape, len) = nested_prepare(v, dtype)?;
        let mut data = AlignedBytes::zeroed(len);
        fill_tensor(v, dtype, &shape, data.as_mut_bytes())?;
        Ok(OwnedDataTensor {
            dtype,
            shape: shape.into_boxed_slice(),
            data,
        })
    }

    /// Owned twin of [`DataTensor::to_nested_in`].
    pub fn to_nested(&self) -> Result<OwnedDataValue, TensorError> {
        emit_nested(self.view(), &OwnedSink)
    }

    /// Owned twin of [`DataTensor::from_json_value_in`]. An
    /// `OwnedDataValue::Tensor` input is deep-copied; match the variant
    /// directly to share its `Arc` instead.
    pub fn from_json_value(v: &OwnedDataValue) -> Result<Self, TensorError> {
        if let OwnedDataValue::Tensor(t) = v {
            return Ok((**t).clone());
        }
        let tagged = tagged_prepare(v)?;
        let mut data = AlignedBytes::zeroed(tagged.byte_len);
        decode_tagged_into(tagged.dtype, tagged.data, data.as_mut_bytes())?;
        Ok(OwnedDataTensor {
            dtype: tagged.dtype,
            shape: tagged.shape.into_boxed_slice(),
            data,
        })
    }
}

impl TryFrom<&OwnedDataValue> for OwnedDataTensor {
    type Error = TensorError;
    fn try_from(v: &OwnedDataValue) -> Result<Self, Self::Error> {
        Self::from_json_value(v)
    }
}

impl fmt::Debug for OwnedDataTensor {
    /// Deliberately lossy: dtype, shape, and byte count, never the payload.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedDataTensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("bytes", &self.data.len)
            .finish()
    }
}

impl PartialEq for OwnedDataTensor {
    /// Structural: dtype, shape, and bytes (bytewise; see [`DataTensor`]).
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.view() == other.view()
    }
}

impl Eq for OwnedDataTensor {}

// ---- Shared walks over both value types ---------------------------------------

/// The slice of each value type the conversions need.
trait Nested: Sized {
    fn items(&self) -> Option<&[Self]>;
    fn leaf_bool(&self) -> Option<bool>;
    fn leaf_number(&self) -> Option<NumberValue>;
    fn leaf_str(&self) -> Option<&str>;
    /// Object entries in order; `None` if not an object.
    fn entries(&self) -> Option<impl ExactSizeIterator<Item = (&str, &Self)>>;
}

impl Nested for DataValue<'_> {
    #[inline]
    fn items(&self) -> Option<&[Self]> {
        self.as_array()
    }
    #[inline]
    fn leaf_bool(&self) -> Option<bool> {
        self.as_bool()
    }
    #[inline]
    fn leaf_number(&self) -> Option<NumberValue> {
        self.as_number().copied()
    }
    #[inline]
    fn leaf_str(&self) -> Option<&str> {
        self.as_str()
    }
    #[inline]
    fn entries(&self) -> Option<impl ExactSizeIterator<Item = (&str, &Self)>> {
        self.as_object().map(|o| o.iter().map(|(k, v)| (*k, v)))
    }
}

impl Nested for OwnedDataValue {
    #[inline]
    fn items(&self) -> Option<&[Self]> {
        self.as_array()
    }
    #[inline]
    fn leaf_bool(&self) -> Option<bool> {
        self.as_bool()
    }
    #[inline]
    fn leaf_number(&self) -> Option<NumberValue> {
        self.as_number().copied()
    }
    #[inline]
    fn leaf_str(&self) -> Option<&str> {
        self.as_str()
    }
    #[inline]
    fn entries(&self) -> Option<impl ExactSizeIterator<Item = (&str, &Self)>> {
        self.as_object()
            .map(|o| o.iter().map(|(k, v)| (k.as_str(), v)))
    }
}

// ---- Nested arrays → tensor -------------------------------------------------

/// Shape from the first leaf's path: descend through index 0 while the
/// value is an array; an empty array contributes a `0` and stops.
fn infer_shape<V: Nested>(v: &V) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut cur = v;
    while let Some(items) = cur.items() {
        shape.push(items.len());
        match items.first() {
            Some(first) => cur = first,
            None => break,
        }
    }
    shape
}

/// Pass 1 of the nested conversion: the inferred shape and the exact byte
/// length to allocate. Structure is checked during the fill, so the tree is
/// walked once.
fn nested_prepare<V: Nested>(v: &V, dtype: DType) -> Result<(Vec<usize>, usize), TensorError> {
    if !dtype.has_native_element() {
        return Err(TensorError::UnsupportedDType(dtype));
    }
    let shape = infer_shape(v);
    let len = expected_byte_len(dtype, &shape)?;
    Ok((shape, len))
}

/// Pass 2: decode every leaf into the zeroed, aligned `out` buffer. One
/// monomorphised walk per dtype; the converter is statically dispatched.
fn fill_tensor<V: Nested>(
    v: &V,
    dtype: DType,
    shape: &[usize],
    out: &mut [u8],
) -> Result<(), TensorError> {
    fn int<V: Nested, T: TryFrom<i64>>(x: &V) -> Option<T> {
        x.leaf_number()
            .and_then(|n| n.as_i64())
            .and_then(|i| T::try_from(i).ok())
    }
    match dtype {
        DType::Bool => fill(v, shape, out, dtype, V::leaf_bool),
        DType::I8 => fill(v, shape, out, dtype, int::<V, i8>),
        DType::U8 => fill(v, shape, out, dtype, int::<V, u8>),
        DType::I16 => fill(v, shape, out, dtype, int::<V, i16>),
        DType::U16 => fill(v, shape, out, dtype, int::<V, u16>),
        DType::I32 => fill(v, shape, out, dtype, int::<V, i32>),
        DType::U32 => fill(v, shape, out, dtype, int::<V, u32>),
        DType::I64 => fill(v, shape, out, dtype, int::<V, i64>),
        DType::U64 => fill(v, shape, out, dtype, |x: &V| {
            x.leaf_number().and_then(|n| n.as_u64())
        }),
        DType::F32 => fill(v, shape, out, dtype, |x: &V| {
            // Rounding within range is inherent; a non-finite result from
            // finite input means the value did not fit, and the crate never
            // manufactures infinity (see `NumberValue::as_i64`).
            x.leaf_number()
                .map(|n| n.as_f64() as f32)
                .filter(|f| f.is_finite())
        }),
        DType::F64 => fill(v, shape, out, dtype, |x: &V| {
            x.leaf_number().map(|n| n.as_f64())
        }),
        // Same rule as `F32`: rounding within range is inherent, but a
        // finite input that lands on ±inf did not fit the dtype.
        #[cfg(feature = "tensor-half")]
        DType::F16 => fill(v, shape, out, dtype, |x: &V| {
            x.leaf_number()
                .map(|n| half::f16::from_f64(n.as_f64()))
                .filter(|f| f.is_finite())
        }),
        #[cfg(feature = "tensor-half")]
        DType::BF16 => fill(v, shape, out, dtype, |x: &V| {
            x.leaf_number()
                .map(|n| half::bf16::from_f64(n.as_f64()))
                .filter(|f| f.is_finite())
        }),
        #[cfg(not(feature = "tensor-half"))]
        DType::F16 | DType::BF16 => Err(TensorError::UnsupportedDType(dtype)),
    }
}

fn fill<V: Nested, T: Element, F: Fn(&V) -> Option<T>>(
    v: &V,
    shape: &[usize],
    out: &mut [u8],
    dtype: DType,
    conv: F,
) -> Result<(), TensorError> {
    let typed = cast_slice_mut::<T>(out);
    let mut next = 0usize;
    fill_rec(v, 0, shape, typed, &mut next, dtype, &conv)?;
    debug_assert_eq!(next, typed.len());
    Ok(())
}

/// One pass over the tree: every array at depth `d` must have length
/// `shape[d]` and every leaf must sit at depth `shape.len()`, checked as
/// the leaves are decoded in row-major order. The innermost dimension is a
/// plain loop rather than a recursion per leaf.
fn fill_rec<V: Nested, T: Element, F: Fn(&V) -> Option<T>>(
    v: &V,
    depth: usize,
    shape: &[usize],
    out: &mut [T],
    next: &mut usize,
    dtype: DType,
    conv: &F,
) -> Result<(), TensorError> {
    let Some(items) = v.items() else {
        if depth != shape.len() {
            return Err(TensorError::Ragged { depth });
        }
        return put(v, out, next, dtype, conv);
    };
    if depth == shape.len() || items.len() != shape[depth] {
        return Err(TensorError::Ragged { depth });
    }
    if depth + 1 == shape.len() {
        for item in items {
            if item.items().is_some() {
                return Err(TensorError::Ragged { depth: depth + 1 });
            }
            put(item, out, next, dtype, conv)?;
        }
    } else {
        for item in items {
            fill_rec(item, depth + 1, shape, out, next, dtype, conv)?;
        }
    }
    Ok(())
}

#[inline]
fn put<V: Nested, T: Element, F: Fn(&V) -> Option<T>>(
    leaf: &V,
    out: &mut [T],
    next: &mut usize,
    dtype: DType,
    conv: &F,
) -> Result<(), TensorError> {
    let index = *next;
    match conv(leaf) {
        Some(x) => {
            out[index] = x;
            *next = index + 1;
            Ok(())
        }
        None => Err(TensorError::Element {
            index,
            expected: dtype,
        }),
    }
}

// ---- Tensor → nested arrays -------------------------------------------------

enum Leaf {
    Bool(bool),
    Num(NumberValue),
}

trait NestedSink {
    type Out;
    fn leaf(&self, leaf: Leaf) -> Self::Out;
    fn array<F: FnMut(usize) -> Self::Out>(&self, len: usize, f: F) -> Self::Out;
}

struct ArenaSink<'a>(&'a Bump);

impl<'a> NestedSink for ArenaSink<'a> {
    type Out = DataValue<'a>;
    #[inline]
    fn leaf(&self, leaf: Leaf) -> DataValue<'a> {
        match leaf {
            Leaf::Bool(b) => DataValue::Bool(b),
            Leaf::Num(n) => DataValue::Number(n),
        }
    }
    #[inline]
    fn array<F: FnMut(usize) -> DataValue<'a>>(&self, len: usize, f: F) -> DataValue<'a> {
        DataValue::Array(self.0.alloc_slice_fill_with(len, f))
    }
}

struct OwnedSink;

impl NestedSink for OwnedSink {
    type Out = OwnedDataValue;
    #[inline]
    fn leaf(&self, leaf: Leaf) -> OwnedDataValue {
        match leaf {
            Leaf::Bool(b) => OwnedDataValue::Bool(b),
            Leaf::Num(n) => OwnedDataValue::Number(n),
        }
    }
    #[inline]
    fn array<F: FnMut(usize) -> OwnedDataValue>(&self, len: usize, f: F) -> OwnedDataValue {
        OwnedDataValue::Array((0..len).map(f).collect())
    }
}

/// One monomorphised walk per dtype; the leaf converter is statically
/// dispatched.
fn emit_nested<S: NestedSink>(t: DataTensor<'_>, sink: &S) -> Result<S::Out, TensorError> {
    fn view<T: Element>(t: DataTensor<'_>) -> &[T] {
        t.as_slice().expect("dtype matches element type")
    }
    fn int<T: Into<i64>>(x: T) -> Leaf {
        Leaf::Num(NumberValue::Integer(x.into()))
    }
    let shape = t.shape();
    Ok(match t.dtype() {
        DType::Bool => emit_rec(shape, view::<bool>(t), sink, &Leaf::Bool),
        DType::I8 => emit_rec(shape, view::<i8>(t), sink, &int::<i8>),
        DType::U8 => emit_rec(shape, view::<u8>(t), sink, &int::<u8>),
        DType::I16 => emit_rec(shape, view::<i16>(t), sink, &int::<i16>),
        DType::U16 => emit_rec(shape, view::<u16>(t), sink, &int::<u16>),
        DType::I32 => emit_rec(shape, view::<i32>(t), sink, &int::<i32>),
        DType::U32 => emit_rec(shape, view::<u32>(t), sink, &int::<u32>),
        DType::I64 => emit_rec(shape, view::<i64>(t), sink, &int::<i64>),
        DType::U64 => emit_rec(shape, view::<u64>(t), sink, &|x: u64| {
            Leaf::Num(NumberValue::from_u64(x))
        }),
        // `from_f64` canonicalises whole floats to Integer, as the parser
        // does; NaN / ±inf stay Float and render as `null`.
        DType::F32 => emit_rec(shape, view::<f32>(t), sink, &|x: f32| {
            Leaf::Num(NumberValue::from_f64(f64::from(x)))
        }),
        DType::F64 => emit_rec(shape, view::<f64>(t), sink, &|x: f64| {
            Leaf::Num(NumberValue::from_f64(x))
        }),
        #[cfg(feature = "tensor-half")]
        DType::F16 => emit_rec(shape, view::<half::f16>(t), sink, &|x: half::f16| {
            Leaf::Num(NumberValue::from_f64(x.to_f64()))
        }),
        #[cfg(feature = "tensor-half")]
        DType::BF16 => emit_rec(shape, view::<half::bf16>(t), sink, &|x: half::bf16| {
            Leaf::Num(NumberValue::from_f64(x.to_f64()))
        }),
        #[cfg(not(feature = "tensor-half"))]
        DType::F16 | DType::BF16 => return Err(TensorError::UnsupportedDType(t.dtype())),
    })
}

fn emit_rec<S: NestedSink, T: Element, L: Fn(T) -> Leaf>(
    shape: &[usize],
    elems: &[T],
    sink: &S,
    leaf: &L,
) -> S::Out {
    let Some((&n, rest)) = shape.split_first() else {
        return sink.leaf(leaf(elems[0]));
    };
    if rest.is_empty() {
        return sink.array(n, |i| sink.leaf(leaf(elems[i])));
    }
    // A zero-length dimension gives `n == 0` or `stride == 0`; either way
    // the closure sees only empty sub-slices or is never called.
    let stride: usize = rest.iter().product();
    sink.array(n, |i| {
        emit_rec(rest, &elems[i * stride..(i + 1) * stride], sink, leaf)
    })
}

// ---- `{"tensor": ...}` boundary decoding -----------------------------------------

/// The wire form, pulled apart and sized *before* anything is allocated or
/// decoded.
struct Tagged<'v> {
    dtype: DType,
    shape: Vec<usize>,
    data: &'v str,
    byte_len: usize,
}

/// Strict: exactly one top-level key, exactly the three inner keys, and a
/// base64 payload whose decoded length matches the shape.
fn tagged_prepare<V: Nested>(v: &V) -> Result<Tagged<'_>, TensorError> {
    let mut top = v.entries().ok_or(TensorError::NotATensor)?;
    if top.len() != 1 {
        return Err(TensorError::NotATensor);
    }
    let Some((key, body)) = top.next() else {
        return Err(TensorError::NotATensor);
    };
    if key != DataTensor::JSON_TAG {
        return Err(TensorError::NotATensor);
    }
    let (mut dtype, mut shape, mut data) = (None, None, None);
    for (k, val) in body.entries().ok_or(TensorError::NotATensor)? {
        // A repeated key would last-win, which is the same silent misreading
        // `UnexpectedField` exists to prevent: a writer that means something
        // by the duplicate must not be read as if it had sent only the last.
        let duplicate = || TensorError::UnexpectedField(k.to_string());
        match k {
            KEY_DTYPE => {
                if dtype.is_some() {
                    return Err(duplicate());
                }
                dtype = Some(
                    val.leaf_str()
                        .ok_or(TensorError::NotATensor)?
                        .parse::<DType>()?,
                );
            }
            KEY_SHAPE => {
                if shape.is_some() {
                    return Err(duplicate());
                }
                // `as_i64` accepts whole floats (`2.0`), matching the crate's
                // number canonicalisation; negatives are refused.
                let dims = val
                    .items()
                    .ok_or(TensorError::NotATensor)?
                    .iter()
                    .map(|item| {
                        item.leaf_number()
                            .and_then(|n| n.as_i64())
                            .and_then(|i| usize::try_from(i).ok())
                            .ok_or(TensorError::NotATensor)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shape = Some(dims);
            }
            KEY_DATA => {
                if data.is_some() {
                    return Err(duplicate());
                }
                data = Some(val.leaf_str().ok_or(TensorError::NotATensor)?);
            }
            other => return Err(TensorError::UnexpectedField(other.to_string())),
        }
    }
    let (Some(dtype), Some(shape), Some(data)) = (dtype, shape, data) else {
        return Err(TensorError::NotATensor);
    };
    let byte_len = expected_byte_len(dtype, &shape)?;
    let actual = base64::decoded_len(data.as_bytes()).ok_or(TensorError::InvalidBase64)?;
    if actual != byte_len {
        return Err(TensorError::LengthMismatch {
            expected: byte_len,
            actual,
        });
    }
    Ok(Tagged {
        dtype,
        shape,
        data,
        byte_len,
    })
}

/// Decode the payload into a buffer sized by [`tagged_prepare`], then apply
/// the payload rule.
fn decode_tagged_into(dtype: DType, b64: &str, out: &mut [u8]) -> Result<(), TensorError> {
    if !base64::decode_into(b64.as_bytes(), out) {
        return Err(TensorError::InvalidBase64);
    }
    validate_payload(dtype, out)
}

// ---- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::collections::Vec as BumpVec;

    #[test]
    fn dtype_names_sizes_round_trip() {
        for d in DType::ALL {
            assert_eq!(DType::from_name(d.name()), Some(d));
            assert_eq!(DType::from_name(&d.name().to_uppercase()), Some(d));
            assert_eq!(d.name().parse::<DType>().unwrap(), d);
            assert_eq!(d.to_string(), d.name());
            assert_eq!(d.byte_len(3), Some(3 * d.size_of()));
            assert_eq!(d.align_of(), d.size_of());
        }
        assert_eq!(DType::from_name("nope"), None);
        assert_eq!(
            "nope".parse::<DType>(),
            Err(TensorError::UnknownDType("nope".into()))
        );
        assert_eq!(DType::F64.byte_len(usize::MAX), None);
    }

    #[test]
    fn from_slice_wraps_bump_vec_without_copy() {
        let arena = Bump::new();
        let mut v: BumpVec<'_, f32> = BumpVec::new_in(&arena);
        v.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let slice = v.into_bump_slice();
        let shape = arena.alloc_slice_copy(&[2, 3]);
        let t = DataTensor::from_slice(shape, slice).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.data().len(), 24);
        let view = t.as_slice::<f32>().unwrap();
        assert_eq!(view, slice);
        assert_eq!(
            view.as_ptr(),
            slice.as_ptr(),
            "typed view must be zero-copy"
        );
        assert!(t.as_slice::<i32>().is_none());
    }

    #[test]
    fn from_bytes_validates_length_alignment_and_bool() {
        #[repr(align(8))]
        struct Aligned([u8; 16]);
        let buf = Aligned([0; 16]);
        let shape = [2usize];

        assert!(DataTensor::from_bytes(DType::F32, &shape, &buf.0[..8]).is_ok());
        assert_eq!(
            DataTensor::from_bytes(DType::F32, &shape, &buf.0[..9]),
            Err(TensorError::LengthMismatch {
                expected: 8,
                actual: 9
            })
        );
        assert_eq!(
            DataTensor::from_bytes(DType::F32, &shape, &buf.0[1..9]),
            Err(TensorError::Misaligned { required: 4 })
        );
        // The copying constructor re-aligns misaligned input.
        let arena = Bump::new();
        let t = DataTensor::from_bytes_in(DType::F32, &shape, &buf.0[1..9], &arena).unwrap();
        assert_eq!(t.data().as_ptr() as usize % 4, 0);

        assert_eq!(
            DataTensor::from_bytes(DType::Bool, &[3], &[0, 1, 2]),
            Err(TensorError::InvalidBool)
        );
        assert_eq!(
            OwnedDataTensor::from_bytes(DType::Bool, [3], &[1, 0, 9]).unwrap_err(),
            TensorError::InvalidBool
        );
        let ok = DataTensor::from_bytes(DType::Bool, &[3], &[1, 0, 1]).unwrap();
        assert_eq!(ok.as_slice::<bool>().unwrap(), &[true, false, true]);
    }

    #[test]
    fn shape_overflow_is_rejected_before_anything_else() {
        assert_eq!(
            DataTensor::from_bytes(DType::U8, &[usize::MAX, 2], &[]),
            Err(TensorError::ShapeOverflow)
        );
        assert_eq!(
            DataTensor::from_bytes(DType::F32, &[usize::MAX / 4 + 1], &[]),
            Err(TensorError::ShapeOverflow)
        );
    }

    #[test]
    fn zero_dim_and_zero_size_shapes() {
        let scalar = DataTensor::from_slice(&[], &[42i32]).unwrap();
        assert_eq!(scalar.numel(), 1);
        assert_eq!(scalar.ndim(), 0);

        let empty = DataTensor::from_slice::<f64>(&[0, 3], &[]).unwrap();
        assert_eq!(empty.numel(), 0);
        assert!(empty.data().is_empty());
        assert_eq!(empty.as_slice::<f64>(), Some(&[][..]));

        assert_eq!(
            DataTensor::from_slice(&[0, 3], &[1.0f64]),
            Err(TensorError::LengthMismatch {
                expected: 0,
                actual: 8
            })
        );
    }

    #[test]
    fn owned_round_trips_and_is_aligned() {
        let arena = Bump::new();
        let t = DataTensor::from_slice_in(&[2, 2], &[1u64, 2, 3, 4], &arena).unwrap();
        let owned = t.to_owned();
        assert_eq!(owned.dtype(), DType::U64);
        assert_eq!(owned.shape(), &[2, 2]);
        assert_eq!(owned.as_slice::<u64>().unwrap(), &[1, 2, 3, 4]);
        assert_eq!(owned.data().as_ptr() as usize % 8, 0);

        let back = owned.to_arena(&arena);
        assert_eq!(back, t);
        assert_eq!(back.to_owned(), owned);

        // Misaligned owned input is copied into aligned storage.
        let raw: Vec<u8> = (0..17).collect();
        let o = OwnedDataTensor::from_bytes(DType::I32, vec![4], &raw[1..17]).unwrap();
        assert_eq!(o.data().as_ptr() as usize % 4, 0);
        assert_eq!(o.data(), &raw[1..17]);
    }

    #[test]
    fn debug_is_bounded() {
        let data = vec![0u8; 40_000];
        let o = OwnedDataTensor::from_bytes(DType::F32, [100, 100], &data).unwrap();
        let s = format!("{o:?}");
        assert!(s.len() < 120, "{s}");
        assert!(s.contains("F32") && s.contains("[100, 100]") && s.contains("40000"));
        let arena = Bump::new();
        let t = o.to_arena(&arena);
        assert!(format!("{t:?}").len() < 120);
    }

    #[test]
    fn equality_is_bytewise() {
        let a = OwnedDataTensor::from_slice([2], &[0.0f32, f32::NAN]).unwrap();
        let b = OwnedDataTensor::from_slice([2], &[0.0f32, f32::NAN]).unwrap();
        let c = OwnedDataTensor::from_slice([2], &[-0.0f32, f32::NAN]).unwrap();
        assert_eq!(a, b, "identical NaN bit patterns are equal");
        assert_ne!(a, c, "0.0 and -0.0 differ bytewise");
        let d = OwnedDataTensor::from_slice([1, 2], &[0.0f32, f32::NAN]).unwrap();
        assert_ne!(a, d, "shape participates");
    }

    // ---- nested conversion ----

    fn parse<'a>(s: &'a str, arena: &'a Bump) -> DataValue<'a> {
        DataValue::from_str(s, arena).unwrap()
    }

    #[test]
    fn nested_shape_inference_table() {
        let arena = Bump::new();
        let cases: &[(&str, Result<&[usize], TensorError>)] = &[
            ("7", Ok(&[])),
            ("[]", Ok(&[0])),
            ("[[]]", Ok(&[1, 0])),
            ("[[1],[2]]", Ok(&[2, 1])),
            ("[[1,2],[3,4],[5,6]]", Ok(&[3, 2])),
            ("[[1,2],[3]]", Err(TensorError::Ragged { depth: 1 })),
            ("[1,[2]]", Err(TensorError::Ragged { depth: 1 })),
            ("[[1],2]", Err(TensorError::Ragged { depth: 1 })),
            ("[[],[1]]", Err(TensorError::Ragged { depth: 1 })),
        ];
        for (input, expected) in cases {
            let v = parse(input, &arena);
            let got = DataTensor::from_nested_in(&v, DType::I32, &arena).map(|t| t.shape());
            assert_eq!(got, *expected, "input {input}");
        }
    }

    #[test]
    fn nested_element_rules() {
        let arena = Bump::new();
        let err = |index, expected| Err(TensorError::Element { index, expected });
        let run = |s: &str, d: DType| {
            let v = parse(s, &arena);
            DataTensor::from_nested_in(&v, d, &arena).map(|t| t.data().to_vec())
        };

        // Bool accepts only Bool: no truthiness.
        assert_eq!(run("[true,false]", DType::Bool), Ok(vec![1, 0]));
        assert_eq!(run("[true,1]", DType::Bool), err(1, DType::Bool));
        // Integers: exact range, whole floats accepted, no truncation.
        assert_eq!(run("[127,-128,2.0]", DType::I8), Ok(vec![127, 0x80, 2]));
        assert_eq!(run("[1,200]", DType::I8), err(1, DType::I8));
        assert_eq!(run("[1,-1]", DType::U8), err(1, DType::U8));
        assert_eq!(run("[[1,2],[3,1.5]]", DType::I32), err(3, DType::I32));
        assert!(run("[2147483647]", DType::I32).is_ok());
        assert_eq!(run("[2147483648]", DType::I32), err(0, DType::I32));
        // u64 above i64::MAX arrives as a Float from the parser.
        let v = parse("[9223372036854775808]", &arena);
        let t = DataTensor::from_nested_in(&v, DType::U64, &arena).unwrap();
        assert_eq!(t.as_slice::<u64>().unwrap(), &[1u64 << 63]);
        assert_eq!(run("[-1]", DType::U64), err(0, DType::U64));
        // Floats: rounding accepted, overflow refused, no manufactured inf.
        assert_eq!(run("[1e39]", DType::F32), err(0, DType::F32));
        let v = parse("[3.4028234663852886e38]", &arena); // f32::MAX as f64
        let t = DataTensor::from_nested_in(&v, DType::F32, &arena).unwrap();
        assert_eq!(t.as_slice::<f32>().unwrap(), &[f32::MAX]);
        let v = parse("[3.4028234663852886e38, 0.1]", &arena);
        let t = DataTensor::from_nested_in(&v, DType::F32, &arena).unwrap();
        assert_eq!(t.as_slice::<f32>().unwrap(), &[f32::MAX, 0.1f32]);
        assert!(run("[1e39]", DType::F64).is_ok());
        // Wrong JSON types, including numeric strings, are refused.
        assert_eq!(run(r#"["42"]"#, DType::I64), err(0, DType::I64));
        assert_eq!(run("[null]", DType::F64), err(0, DType::F64));
        assert_eq!(run(r#"[{"a":1}]"#, DType::F64), err(0, DType::F64));
        // F16 / BF16 need `tensor-half`; without it they are bytes only.
        #[cfg(not(feature = "tensor-half"))]
        {
            assert_eq!(
                run("[1]", DType::F16),
                Err(TensorError::UnsupportedDType(DType::F16))
            );
            assert_eq!(
                run("[1]", DType::BF16),
                Err(TensorError::UnsupportedDType(DType::BF16))
            );
        }
        // With it they follow the same float rule at their own range: f16
        // tops out at 65504, bf16 keeps f32's exponent.
        #[cfg(feature = "tensor-half")]
        {
            assert_eq!(run("[1,-2.5]", DType::F16), Ok(vec![0, 0x3C, 0, 0xC1]));
            assert_eq!(run("[65504]", DType::F16), Ok(vec![0xFF, 0x7B]));
            assert_eq!(run("[70000]", DType::F16), err(0, DType::F16));
            assert!(run("[70000]", DType::BF16).is_ok());
            assert_eq!(run("[1e39]", DType::BF16), err(0, DType::BF16));
            assert_eq!(run(r#"["1"]"#, DType::F16), err(0, DType::F16));
        }
    }

    #[test]
    fn value_that_rounds_down_to_f32_max_is_accepted() {
        // f32::MAX + 1e30 is below the rounding midpoint to 2^128.
        let owned = OwnedDataValue::Array(vec![OwnedDataValue::from_f64(f32::MAX as f64 + 1e30)]);
        let t = OwnedDataTensor::from_nested(&owned, DType::F32).unwrap();
        assert_eq!(t.as_slice::<f32>().unwrap(), &[f32::MAX]);
    }

    #[test]
    fn nested_round_trip_every_native_dtype() {
        let arena = Bump::new();
        let inputs: &[(&str, DType)] = &[
            ("[[true,false],[false,true]]", DType::Bool),
            ("[[-1,2],[3,-4]]", DType::I8),
            ("[[1,2],[3,255]]", DType::U8),
            ("[[-300,2],[3,4]]", DType::I16),
            ("[[65535,2],[3,4]]", DType::U16),
            ("[[-70000,2],[3,4]]", DType::I32),
            ("[[4000000000,2],[3,4]]", DType::U32),
            ("[[-9223372036854775808,2],[3,4]]", DType::I64),
            ("[[9223372036854775807,2],[3,4]]", DType::U64),
            ("[[1.5,2],[3,-0.25]]", DType::F32),
            ("[[1.5,2],[3,1e300]]", DType::F64),
            ("7", DType::I32),
            ("[]", DType::F32),
            ("[[]]", DType::U8),
        ];
        for (input, dtype) in inputs {
            let v = parse(input, &arena);
            let t = DataTensor::from_nested_in(&v, *dtype, &arena).unwrap();
            let back = t.to_nested_in(&arena).unwrap();
            assert_eq!(back, v, "arena round trip for {input} as {dtype}");

            let owned = v.to_owned();
            let ot = OwnedDataTensor::from_nested(&owned, *dtype).unwrap();
            assert_eq!(ot, t.to_owned());
            assert_eq!(
                ot.to_nested().unwrap(),
                owned,
                "owned round trip for {input}"
            );
        }
    }

    #[test]
    fn u64_above_i64_max_emits_float_and_reads_back() {
        let arena = Bump::new();
        let t = DataTensor::from_slice_in(&[1], &[1u64 << 63], &arena).unwrap();
        let v = t.to_nested_in(&arena).unwrap();
        assert!(v[0].is_f64());
        let back = DataTensor::from_nested_in(&v, DType::U64, &arena).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn to_nested_canonicalises_like_the_parser() {
        let arena = Bump::new();
        let t = DataTensor::from_slice_in(&[3], &[2.0f32, 0.5, f32::NAN], &arena).unwrap();
        let v = t.to_nested_in(&arena).unwrap();
        assert!(v[0].is_i64(), "whole float collapses to Integer");
        assert_eq!(v[1].as_f64(), Some(0.5));
        assert!(v[2].as_f64().unwrap().is_nan());
        assert_eq!(v.to_string(), "[2,0.5,null]");
        assert_eq!(
            DataTensor::from_slice_in(&[1], &[0u16], &arena)
                .unwrap()
                .dtype(),
            DType::U16
        );
        let f16 = DataTensor::from_bytes(DType::F16, &[1], &[0, 0]).unwrap();
        #[cfg(not(feature = "tensor-half"))]
        assert_eq!(
            f16.to_nested_in(&arena),
            Err(TensorError::UnsupportedDType(DType::F16))
        );
        // Under `tensor-half` it expands, canonicalising whole floats the
        // same way every other float dtype does.
        #[cfg(feature = "tensor-half")]
        assert_eq!(f16.to_nested_in(&arena).unwrap().to_string(), "[0]");
    }

    #[test]
    fn deep_nesting_near_parser_limit() {
        let arena = Bump::new();
        let depth = 250;
        let s = format!("{}1{}", "[".repeat(depth), "]".repeat(depth));
        let v = parse(&s, &arena);
        let t = DataTensor::from_nested_in(&v, DType::I64, &arena).unwrap();
        assert_eq!(t.ndim(), depth);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.to_nested_in(&arena).unwrap(), v);
    }

    // ---- boundary decoder ----

    #[test]
    fn tagged_form_round_trips_every_dtype() {
        let arena = Bump::new();
        for d in DType::ALL {
            let bytes: Vec<u8> = (0..d.size_of() * 6).map(|i| (i % 2) as u8).collect();
            let t = DataTensor::from_bytes_in(d, &[2, 3], &bytes, &arena).unwrap();
            let json = DataValue::tensor(arena.alloc(t)).to_string();
            let parsed = parse(&json, &arena);
            assert!(parsed.is_object(), "parser must not produce Tensor");
            let back = DataTensor::from_json_value_in(&parsed, &arena).unwrap();
            assert_eq!(back, t, "dtype {d}");

            let owned = OwnedDataTensor::from_json_value(&parsed.to_owned()).unwrap();
            assert_eq!(owned, t.to_owned());
            let via_try: OwnedDataTensor = (&parsed.to_owned()).try_into().unwrap();
            assert_eq!(via_try, owned);
        }
    }

    #[test]
    fn tagged_form_passes_a_tensor_value_through() {
        let arena = Bump::new();
        let t = DataTensor::from_slice_in(&[2], &[1i8, 2], &arena).unwrap();
        let v = DataValue::tensor_in(t, &arena);
        let back = DataTensor::from_json_value_in(&v, &arena).unwrap();
        assert_eq!(back.data().as_ptr(), t.data().as_ptr(), "no copy");
        let ov = OwnedDataValue::tensor(t.to_owned());
        assert_eq!(OwnedDataTensor::from_json_value(&ov).unwrap(), t.to_owned());
    }

    #[test]
    fn tagged_form_errors() {
        let arena = Bump::new();
        let dec = |s: &str| {
            let v = parse(s, &arena);
            DataTensor::from_json_value_in(&v, &arena).map(|t| t.to_owned())
        };
        assert_eq!(dec(r#"{"a":1}"#), Err(TensorError::NotATensor));
        assert_eq!(dec(r#"[1,2]"#), Err(TensorError::NotATensor));
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[1],"data":"AA=="},"x":1}"#),
            Err(TensorError::NotATensor)
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[1]}}"#),
            Err(TensorError::NotATensor)
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[1],"data":"AA==","strides":[1]}}"#),
            Err(TensorError::UnexpectedField("strides".into()))
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"f7","shape":[1],"data":"AA=="}}"#),
            Err(TensorError::UnknownDType("f7".into()))
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[1],"data":"A"}}"#),
            Err(TensorError::InvalidBase64)
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[1],"data":"A!=="}}"#),
            Err(TensorError::InvalidBase64)
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[2],"data":"AA=="}}"#),
            Err(TensorError::LengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[9223372036854775807,4],"data":""}}"#),
            Err(TensorError::ShapeOverflow)
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"bool","shape":[1],"data":"Ag=="}}"#),
            Err(TensorError::InvalidBool)
        );
        // Shape entries: whole floats accepted, negatives / fractions refused.
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[2.0],"data":"AAE="}}"#)
                .unwrap()
                .shape(),
            &[2]
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[-1],"data":""}}"#),
            Err(TensorError::NotATensor)
        );
        assert_eq!(
            dec(r#"{"tensor":{"dtype":"u8","shape":[1.5],"data":""}}"#),
            Err(TensorError::NotATensor)
        );
        // Key order inside tensor does not matter; casing of dtype does not.
        let t = dec(r#"{"tensor":{"data":"AAE=","shape":[2],"dtype":"U8"}}"#).unwrap();
        assert_eq!(t.as_slice::<u8>().unwrap(), &[0, 1]);
    }

    #[test]
    fn error_display_is_informative() {
        let e = TensorError::Element {
            index: 3,
            expected: DType::I32,
        };
        assert_eq!(e.to_string(), "element 3 is not representable as i32");
        let e: Box<dyn std::error::Error> = Box::new(TensorError::ShapeOverflow);
        assert!(e.to_string().contains("overflow"));
    }
}
