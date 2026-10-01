//! [`OwnedDataValue`] — heap-owned counterpart to [`DataValue`].
//!
//! Use this when a value must outlive its arena: long-lived caches,
//! function return values across arena boundaries, results stored in
//! global state, etc. Construction goes through the same fast hand-rolled
//! parser via a throwaway arena and a deep-clone out of it.
//!
//! For hot-path workloads keep using [`DataValue`] — the owned form is
//! strictly slower (heap allocation per composite node) and exists only
//! to escape the arena lifetime when needed.

use core::ops::Index;

use bumpalo::Bump;

#[cfg(feature = "datetime")]
use crate::datetime::{DataDateTime, DataDuration};
use crate::number::NumberValue;
use crate::parser::ParseError;
#[cfg(feature = "tensor")]
use crate::tensor::OwnedDataTensor;
use crate::value::DataValue;
#[cfg(feature = "tensor")]
use std::sync::Arc;

/// Heap-owned JSON value tree. Variants mirror [`DataValue`] one-for-one;
/// no lifetime parameter.
#[derive(Debug, Clone, Default)]
pub enum OwnedDataValue {
    #[default]
    Null,
    Bool(bool),
    Number(NumberValue),
    String(String),
    Array(Vec<OwnedDataValue>),
    Object(Vec<(String, OwnedDataValue)>),
    #[cfg(feature = "datetime")]
    DateTime(DataDateTime),
    #[cfg(feature = "datetime")]
    Duration(DataDuration),
    /// Opaque n-dimensional typed buffer. The one shared-ownership payload:
    /// the buffer is immutable and potentially large, so clones of the tree
    /// share it instead of deep-copying. Keeps the enum at 32 bytes.
    #[cfg(feature = "tensor")]
    Tensor(Arc<OwnedDataTensor>),
}

static OWNED_NULL: OwnedDataValue = OwnedDataValue::Null;

// Feature-gated variants must not grow the enum past the 24-byte Vec/String
// payload plus tag. Checked on 64-bit targets.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<OwnedDataValue>() == 32);

impl core::str::FromStr for OwnedDataValue {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let arena = Bump::new();
        let v = DataValue::from_str(s, &arena)?;
        Ok(v.to_owned())
    }
}

impl OwnedDataValue {
    // ---- Construction ----

    /// Parse JSON into an `OwnedDataValue`. Internally parses into a
    /// throwaway arena (using the fast hand-rolled parser) and deep-clones
    /// the result out — so JSON parsing speed matches `DataValue::from_str`,
    /// minus the deep-clone tail.
    ///
    /// Also available via the [`std::str::FromStr`] trait.
    pub fn from_json(input: &str) -> Result<Self, ParseError> {
        input.parse()
    }

    // ---- Type predicates ----

    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, OwnedDataValue::Null)
    }
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(self, OwnedDataValue::Bool(_))
    }
    #[inline]
    pub fn is_number(&self) -> bool {
        matches!(self, OwnedDataValue::Number(_))
    }
    #[inline]
    pub fn is_i64(&self) -> bool {
        matches!(self, OwnedDataValue::Number(NumberValue::Integer(_)))
    }
    #[inline]
    pub fn is_f64(&self) -> bool {
        matches!(self, OwnedDataValue::Number(NumberValue::Float(_)))
    }
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, OwnedDataValue::String(_))
    }
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, OwnedDataValue::Array(_))
    }
    #[inline]
    pub fn is_object(&self) -> bool {
        matches!(self, OwnedDataValue::Object(_))
    }
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn is_datetime(&self) -> bool {
        matches!(self, OwnedDataValue::DateTime(_))
    }
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn is_duration(&self) -> bool {
        matches!(self, OwnedDataValue::Duration(_))
    }
    #[cfg(feature = "tensor")]
    #[inline]
    pub fn is_tensor(&self) -> bool {
        matches!(self, OwnedDataValue::Tensor(_))
    }

    // ---- Accessors ----

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            OwnedDataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            OwnedDataValue::Number(n) => n.as_i64(),
            _ => None,
        }
    }
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            OwnedDataValue::Number(n) => Some(n.as_f64()),
            _ => None,
        }
    }
    #[inline]
    pub fn as_number(&self) -> Option<&NumberValue> {
        match self {
            OwnedDataValue::Number(n) => Some(n),
            _ => None,
        }
    }
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            OwnedDataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    #[inline]
    pub fn as_array(&self) -> Option<&[OwnedDataValue]> {
        match self {
            OwnedDataValue::Array(items) => Some(items.as_slice()),
            _ => None,
        }
    }
    #[inline]
    pub fn as_object(&self) -> Option<&[(String, OwnedDataValue)]> {
        match self {
            OwnedDataValue::Object(pairs) => Some(pairs.as_slice()),
            _ => None,
        }
    }
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn as_datetime(&self) -> Option<&DataDateTime> {
        match self {
            OwnedDataValue::DateTime(d) => Some(d),
            _ => None,
        }
    }
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn as_duration(&self) -> Option<&DataDuration> {
        match self {
            OwnedDataValue::Duration(d) => Some(d),
            _ => None,
        }
    }
    #[cfg(feature = "tensor")]
    #[inline]
    pub fn as_tensor(&self) -> Option<&OwnedDataTensor> {
        match self {
            OwnedDataValue::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Wrap a tensor in an `Arc` and hold it. To share an existing `Arc`,
    /// use `OwnedDataValue::from(arc)`.
    #[cfg(feature = "tensor")]
    #[inline]
    pub fn tensor(t: OwnedDataTensor) -> Self {
        OwnedDataValue::Tensor(Arc::new(t))
    }

    /// `serde_json::Value::get`-style lookup.
    #[inline]
    pub fn get<I: OwnedValueIndex>(&self, index: I) -> Option<&OwnedDataValue> {
        I::index_into(&index, self)
    }

    #[inline]
    pub fn len(&self) -> Option<usize> {
        match self {
            OwnedDataValue::Array(a) => Some(a.len()),
            OwnedDataValue::Object(o) => Some(o.len()),
            _ => None,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|n| n == 0)
    }

    /// Iterate array items. Returns an empty iterator if `self` is not an
    /// array.
    #[inline]
    pub fn members(&self) -> core::slice::Iter<'_, OwnedDataValue> {
        match self {
            OwnedDataValue::Array(items) => items.iter(),
            _ => [].iter(),
        }
    }

    /// Iterate object entries as `(key, value)` pairs in insertion order.
    /// Returns an empty iterator if `self` is not an object.
    #[inline]
    pub fn entries(&self) -> OwnedEntriesIter<'_> {
        match self {
            OwnedDataValue::Object(pairs) => OwnedEntriesIter {
                inner: pairs.iter(),
            },
            _ => OwnedEntriesIter { inner: [].iter() },
        }
    }

    /// Serialise to a compact JSON string. Equivalent to `format!("{self}")` /
    /// `self.to_string()` — provided as the conventional name people reach
    /// for, and so callers don't have to import `std::fmt::Write`.
    #[inline]
    pub fn to_json_string(&self) -> String {
        self.to_string()
    }

    /// Deep-copy this owned tree into the given arena. Strings, keys, and
    /// tensor shape and bytes are arena-allocated copies, so the result does
    /// not borrow `self`: the owned tree may be mutated or dropped while the
    /// view is alive. When it is only read, [`view_in`](Self::view_in)
    /// borrows the leaves instead.
    ///
    /// Array/Object use `alloc_slice_fill_with` rather than
    /// `bumpalo::Vec::with_capacity_in` + push: one pre-sized arena
    /// allocation and a tight write loop, skipping the Vec wrapper's
    /// per-push capacity check and the `Layout::array` validation that
    /// `RawVec::allocate_in` re-runs for each nested allocation.
    pub fn to_arena<'a>(&self, arena: &'a Bump) -> DataValue<'a> {
        match self {
            OwnedDataValue::Null => DataValue::Null,
            OwnedDataValue::Bool(b) => DataValue::Bool(*b),
            OwnedDataValue::Number(n) => DataValue::Number(*n),
            OwnedDataValue::String(s) => DataValue::String(arena.alloc_str(s)),
            OwnedDataValue::Array(items) => {
                let slice = arena.alloc_slice_fill_with(items.len(), |i| items[i].to_arena(arena));
                DataValue::Array(slice)
            }
            OwnedDataValue::Object(pairs) => {
                let slice = arena.alloc_slice_fill_with(pairs.len(), |i| {
                    let (k, v) = &pairs[i];
                    (arena.alloc_str(k) as &str, v.to_arena(arena))
                });
                DataValue::Object(slice)
            }
            #[cfg(feature = "datetime")]
            OwnedDataValue::DateTime(d) => DataValue::DateTime(*d),
            #[cfg(feature = "datetime")]
            OwnedDataValue::Duration(d) => DataValue::Duration(*d),
            #[cfg(feature = "tensor")]
            OwnedDataValue::Tensor(t) => DataValue::tensor_in(t.to_arena(arena), arena),
        }
    }

    /// Arena view that borrows from `self`: strings, object keys, and tensor
    /// shape and bytes point into the owned tree. Only the array/object
    /// slices are allocated, plus one 40-byte header per tensor (what
    /// [`DataValue::Tensor`] points at). Cost is proportional to the number
    /// of containers, not the number of string or tensor bytes.
    ///
    /// `self` stays immutably borrowed for as long as the view is alive. Use
    /// [`to_arena`](Self::to_arena) when the owned tree must be mutated or
    /// dropped while the view exists.
    ///
    /// ```compile_fail
    /// # use bumpalo::Bump;
    /// # use datavalue_rs::OwnedDataValue;
    /// let mut owned: OwnedDataValue = r#"{"k":"v"}"#.parse().unwrap();
    /// let arena = Bump::new();
    /// let view = owned.view_in(&arena);
    /// owned = OwnedDataValue::Null; // borrowed by `view`
    /// assert!(view.is_object());
    /// ```
    pub fn view_in<'a>(&'a self, arena: &'a Bump) -> DataValue<'a> {
        match self {
            OwnedDataValue::Null => DataValue::Null,
            OwnedDataValue::Bool(b) => DataValue::Bool(*b),
            OwnedDataValue::Number(n) => DataValue::Number(*n),
            OwnedDataValue::String(s) => DataValue::String(s),
            OwnedDataValue::Array(items) => {
                let slice = arena.alloc_slice_fill_with(items.len(), |i| items[i].view_in(arena));
                DataValue::Array(slice)
            }
            OwnedDataValue::Object(pairs) => {
                let slice = arena.alloc_slice_fill_with(pairs.len(), |i| {
                    let (k, v) = &pairs[i];
                    (k.as_str(), v.view_in(arena))
                });
                DataValue::Object(slice)
            }
            #[cfg(feature = "datetime")]
            OwnedDataValue::DateTime(d) => DataValue::DateTime(*d),
            #[cfg(feature = "datetime")]
            OwnedDataValue::Duration(d) => DataValue::Duration(*d),
            // Owned storage already satisfies every tensor invariant
            // (8-byte alignment covers every dtype), so the view is the
            // borrowed form as-is.
            #[cfg(feature = "tensor")]
            OwnedDataValue::Tensor(t) => DataValue::tensor_in(t.view(), arena),
        }
    }
}

impl<'a> DataValue<'a> {
    /// Deep-clone this arena-bound tree into an [`OwnedDataValue`] that
    /// no longer references the arena.
    pub fn to_owned(&self) -> OwnedDataValue {
        match *self {
            DataValue::Null => OwnedDataValue::Null,
            DataValue::Bool(b) => OwnedDataValue::Bool(b),
            DataValue::Number(n) => OwnedDataValue::Number(n),
            DataValue::String(s) => OwnedDataValue::String(s.to_string()),
            DataValue::Array(items) => {
                OwnedDataValue::Array(items.iter().map(DataValue::to_owned).collect())
            }
            DataValue::Object(pairs) => OwnedDataValue::Object(
                pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), v.to_owned()))
                    .collect(),
            ),
            #[cfg(feature = "datetime")]
            DataValue::DateTime(d) => OwnedDataValue::DateTime(d),
            #[cfg(feature = "datetime")]
            DataValue::Duration(d) => OwnedDataValue::Duration(d),
            #[cfg(feature = "tensor")]
            DataValue::Tensor(t) => OwnedDataValue::tensor(t.to_owned()),
        }
    }
}

/// Iterator over `(key, value)` pairs in an [`OwnedDataValue::Object`].
/// Created via [`OwnedDataValue::entries`].
pub struct OwnedEntriesIter<'v> {
    inner: core::slice::Iter<'v, (String, OwnedDataValue)>,
}

impl<'v> Iterator for OwnedEntriesIter<'v> {
    type Item = (&'v str, &'v OwnedDataValue);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k.as_str(), v))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for OwnedEntriesIter<'_> {}

impl PartialEq for OwnedDataValue {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (OwnedDataValue::Null, OwnedDataValue::Null) => true,
            (OwnedDataValue::Bool(a), OwnedDataValue::Bool(b)) => a == b,
            (OwnedDataValue::Number(a), OwnedDataValue::Number(b)) => a == b,
            (OwnedDataValue::String(a), OwnedDataValue::String(b)) => a == b,
            (OwnedDataValue::Array(a), OwnedDataValue::Array(b)) => a == b,
            (OwnedDataValue::Object(a), OwnedDataValue::Object(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter().all(|(k, v)| {
                    b.iter()
                        .find(|(bk, _)| bk == k)
                        .is_some_and(|(_, bv)| v == bv)
                })
            }
            #[cfg(feature = "datetime")]
            (OwnedDataValue::DateTime(a), OwnedDataValue::DateTime(b)) => a == b,
            #[cfg(feature = "datetime")]
            (OwnedDataValue::Duration(a), OwnedDataValue::Duration(b)) => a == b,
            #[cfg(feature = "tensor")]
            (OwnedDataValue::Tensor(a), OwnedDataValue::Tensor(b)) => a == b,
            _ => false,
        }
    }
}

// ---- Index trait dispatch (parallel to ValueIndex for borrowed side) ----

pub trait OwnedValueIndex: private::Sealed {
    fn index_into<'v>(&self, value: &'v OwnedDataValue) -> Option<&'v OwnedDataValue>;
    fn index_into_or_null<'v>(&self, value: &'v OwnedDataValue) -> &'v OwnedDataValue;
}

mod private {
    pub trait Sealed {}
    impl Sealed for str {}
    impl Sealed for String {}
    impl Sealed for usize {}
    impl<T: Sealed + ?Sized> Sealed for &T {}
}

impl OwnedValueIndex for str {
    #[inline]
    fn index_into<'v>(&self, value: &'v OwnedDataValue) -> Option<&'v OwnedDataValue> {
        match value {
            OwnedDataValue::Object(pairs) => pairs.iter().find(|(k, _)| k == self).map(|(_, v)| v),
            _ => None,
        }
    }
    #[inline]
    fn index_into_or_null<'v>(&self, value: &'v OwnedDataValue) -> &'v OwnedDataValue {
        self.index_into(value).unwrap_or(&OWNED_NULL)
    }
}

impl OwnedValueIndex for String {
    #[inline]
    fn index_into<'v>(&self, value: &'v OwnedDataValue) -> Option<&'v OwnedDataValue> {
        self.as_str().index_into(value)
    }
    #[inline]
    fn index_into_or_null<'v>(&self, value: &'v OwnedDataValue) -> &'v OwnedDataValue {
        self.as_str().index_into_or_null(value)
    }
}

impl OwnedValueIndex for usize {
    #[inline]
    fn index_into<'v>(&self, value: &'v OwnedDataValue) -> Option<&'v OwnedDataValue> {
        match value {
            OwnedDataValue::Array(items) => items.get(*self),
            _ => None,
        }
    }
    #[inline]
    fn index_into_or_null<'v>(&self, value: &'v OwnedDataValue) -> &'v OwnedDataValue {
        self.index_into(value).unwrap_or(&OWNED_NULL)
    }
}

impl<T: OwnedValueIndex + ?Sized> OwnedValueIndex for &T {
    #[inline]
    fn index_into<'v>(&self, value: &'v OwnedDataValue) -> Option<&'v OwnedDataValue> {
        (**self).index_into(value)
    }
    #[inline]
    fn index_into_or_null<'v>(&self, value: &'v OwnedDataValue) -> &'v OwnedDataValue {
        (**self).index_into_or_null(value)
    }
}

impl<I: OwnedValueIndex> Index<I> for OwnedDataValue {
    type Output = OwnedDataValue;
    #[inline]
    fn index(&self, index: I) -> &OwnedDataValue {
        index.index_into_or_null(self)
    }
}

// ---- Convenience constructors ----

impl OwnedDataValue {
    #[inline]
    pub fn from_i64(i: i64) -> Self {
        OwnedDataValue::Number(NumberValue::Integer(i))
    }
    #[inline]
    pub fn from_f64(f: f64) -> Self {
        OwnedDataValue::Number(NumberValue::from_f64(f))
    }

    /// Build an `OwnedDataValue::Array` from anything iterable into values
    /// that already convert into `OwnedDataValue` (covers all the existing
    /// `From<bool|i64|f64|String|&str|...>` impls).
    ///
    /// ```
    /// use datavalue_rs::OwnedDataValue;
    /// let v = OwnedDataValue::array([1, 2, 3]);
    /// assert_eq!(v[0].as_i64(), Some(1));
    /// let v = OwnedDataValue::array(["a", "b"]);
    /// assert_eq!(v[1].as_str(), Some("b"));
    /// ```
    pub fn array<I, V>(items: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<OwnedDataValue>,
    {
        OwnedDataValue::Array(items.into_iter().map(Into::into).collect())
    }

    /// Build an `OwnedDataValue::Object` from `(key, value)` pairs.
    ///
    /// ```
    /// use datavalue_rs::OwnedDataValue;
    /// let v = OwnedDataValue::object([("type", "NaN"), ("op", "+")]);
    /// assert_eq!(v["type"].as_str(), Some("NaN"));
    /// let v = OwnedDataValue::object([("count", 42)]);
    /// assert_eq!(v["count"].as_i64(), Some(42));
    /// ```
    pub fn object<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<OwnedDataValue>,
    {
        OwnedDataValue::Object(
            pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }
}

impl From<Vec<(String, OwnedDataValue)>> for OwnedDataValue {
    #[inline]
    fn from(pairs: Vec<(String, OwnedDataValue)>) -> Self {
        OwnedDataValue::Object(pairs)
    }
}

#[cfg(feature = "tensor")]
impl From<OwnedDataTensor> for OwnedDataValue {
    #[inline]
    fn from(t: OwnedDataTensor) -> Self {
        OwnedDataValue::tensor(t)
    }
}

#[cfg(feature = "tensor")]
impl From<Arc<OwnedDataTensor>> for OwnedDataValue {
    /// Share an existing tensor: no copy, one refcount increment.
    #[inline]
    fn from(t: Arc<OwnedDataTensor>) -> Self {
        OwnedDataValue::Tensor(t)
    }
}

impl From<bool> for OwnedDataValue {
    #[inline]
    fn from(b: bool) -> Self {
        OwnedDataValue::Bool(b)
    }
}

macro_rules! from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for OwnedDataValue {
            #[inline]
            fn from(v: $t) -> Self { OwnedDataValue::from_i64(v as i64) }
        }
    )*};
}
from_int!(i8, i16, i32, i64, u8, u16, u32);

impl From<u64> for OwnedDataValue {
    /// Values up to `i64::MAX` stay on the integer path; larger values
    /// fall back to `f64` (matches the parser / serde visitor behaviour).
    #[inline]
    fn from(v: u64) -> Self {
        OwnedDataValue::Number(NumberValue::from_u64(v))
    }
}
impl From<usize> for OwnedDataValue {
    #[inline]
    fn from(v: usize) -> Self {
        OwnedDataValue::from(v as u64)
    }
}
impl From<isize> for OwnedDataValue {
    #[inline]
    fn from(v: isize) -> Self {
        OwnedDataValue::from_i64(v as i64)
    }
}

impl From<f32> for OwnedDataValue {
    #[inline]
    fn from(v: f32) -> Self {
        OwnedDataValue::from_f64(v as f64)
    }
}
impl From<f64> for OwnedDataValue {
    #[inline]
    fn from(v: f64) -> Self {
        OwnedDataValue::from_f64(v)
    }
}

impl From<String> for OwnedDataValue {
    #[inline]
    fn from(s: String) -> Self {
        OwnedDataValue::String(s)
    }
}
impl From<&str> for OwnedDataValue {
    #[inline]
    fn from(s: &str) -> Self {
        OwnedDataValue::String(s.to_string())
    }
}
impl From<&String> for OwnedDataValue {
    #[inline]
    fn from(s: &String) -> Self {
        OwnedDataValue::String(s.clone())
    }
}
impl From<std::borrow::Cow<'_, str>> for OwnedDataValue {
    #[inline]
    fn from(s: std::borrow::Cow<'_, str>) -> Self {
        OwnedDataValue::String(s.into_owned())
    }
}

impl From<()> for OwnedDataValue {
    #[inline]
    fn from(_: ()) -> Self {
        OwnedDataValue::Null
    }
}

impl<T: Into<OwnedDataValue>> From<Option<T>> for OwnedDataValue {
    #[inline]
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => OwnedDataValue::Null,
        }
    }
}

impl<T: Into<OwnedDataValue>> From<Vec<T>> for OwnedDataValue {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        OwnedDataValue::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<OwnedDataValue> + Clone> From<&[T]> for OwnedDataValue {
    #[inline]
    fn from(v: &[T]) -> Self {
        OwnedDataValue::Array(v.iter().cloned().map(Into::into).collect())
    }
}

impl<T: Into<OwnedDataValue>, const N: usize> From<[T; N]> for OwnedDataValue {
    #[inline]
    fn from(v: [T; N]) -> Self {
        OwnedDataValue::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<K: Into<String>, V: Into<OwnedDataValue>> From<std::collections::HashMap<K, V>>
    for OwnedDataValue
{
    /// Note: `HashMap` iteration order is unspecified, so the resulting
    /// `Object` has unspecified key order. Equality is by key set, so this
    /// still round-trips correctly through `PartialEq`.
    #[inline]
    fn from(m: std::collections::HashMap<K, V>) -> Self {
        OwnedDataValue::Object(m.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
    }
}

impl<K: Into<String>, V: Into<OwnedDataValue>> From<std::collections::BTreeMap<K, V>>
    for OwnedDataValue
{
    #[inline]
    fn from(m: std::collections::BTreeMap<K, V>) -> Self {
        OwnedDataValue::Object(m.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip_via_owned() {
        let v = OwnedDataValue::from_json(r#"{"a":1,"b":[true,null,"x"]}"#).unwrap();
        assert_eq!(v["a"].as_i64(), Some(1));
        assert_eq!(v["b"][0].as_bool(), Some(true));
        assert!(v["b"][1].is_null());
        assert_eq!(v["b"][2].as_str(), Some("x"));
    }

    #[test]
    fn arena_to_owned_to_arena_round_trip() {
        let arena = Bump::new();
        let original =
            DataValue::from_str(r#"{"x":42,"y":[1,2,3],"z":{"k":true}}"#, &arena).unwrap();
        let owned = original.to_owned();

        // Drop the arena; owned should still work.
        drop(arena);
        assert_eq!(owned["x"].as_i64(), Some(42));
        assert_eq!(owned["y"][1].as_i64(), Some(2));
        assert_eq!(owned["z"]["k"].as_bool(), Some(true));

        // Rehydrate into a fresh arena and ensure equality on each side.
        let arena2 = Bump::new();
        let back = owned.to_arena(&arena2);
        assert_eq!(back["x"].as_i64(), Some(42));
        assert_eq!(back["y"][1].as_i64(), Some(2));
        assert_eq!(back["z"]["k"].as_bool(), Some(true));

        // And owned -> owned through an arena should equal the original.
        assert_eq!(back.to_owned(), owned);
    }

    /// Arena bytes consumed by `f`, measured on a single pre-sized chunk.
    fn arena_bytes_used(f: impl FnOnce(&Bump)) -> usize {
        let arena = Bump::with_capacity(1 << 20);
        let before = arena.chunk_capacity();
        f(&arena);
        before - arena.chunk_capacity()
    }

    #[test]
    fn view_in_matches_to_arena() {
        let owned = OwnedDataValue::from_json(
            r#"{"s":"plain","esc":"a\"bé","n":[1,-2.5,1e300,18446744073709551616],
                "nested":{"deep":[{"k":null},[],{}],"t":true},"empty":""}"#,
        )
        .unwrap();
        let a1 = Bump::new();
        let a2 = Bump::new();
        let view = owned.view_in(&a1);
        assert_eq!(view, owned.to_arena(&a2));
        assert_eq!(view.to_owned(), owned);
    }

    #[test]
    fn view_in_borrows_strings_and_keys() {
        let owned = OwnedDataValue::from_json(r#"{"key":"value","list":["x"]}"#).unwrap();
        let arena = Bump::new();
        let view = owned.view_in(&arena);

        let (owned_key, owned_val) = owned.entries().next().unwrap();
        let (view_key, view_val) = view.as_object().unwrap()[0];
        assert_eq!(view_key.as_ptr(), owned_key.as_ptr());
        assert_eq!(
            view_val.as_str().unwrap().as_ptr(),
            owned_val.as_str().unwrap().as_ptr()
        );
        assert_eq!(
            view["list"][0].as_str().unwrap().as_ptr(),
            owned["list"][0].as_str().unwrap().as_ptr()
        );

        // Only the two object slots and the one array slot land in the arena.
        let used = arena_bytes_used(|a| {
            owned.view_in(a);
        });
        let slots = 2 * size_of::<(&str, DataValue<'_>)>() + size_of::<DataValue<'_>>();
        assert!(used <= slots + 16, "used {used}, expected ~{slots}");
        assert!(
            used < arena_bytes_used(|a| {
                owned.to_arena(a);
            })
        );
    }

    #[test]
    fn missing_index_returns_null() {
        let v = OwnedDataValue::from_json(r#"{"a":1}"#).unwrap();
        assert!(v["missing"].is_null());
        assert!(v["a"][99].is_null());
    }

    #[test]
    fn equality_object_order_insensitive() {
        let a = OwnedDataValue::Object(vec![
            ("x".to_string(), OwnedDataValue::from_i64(1)),
            ("y".to_string(), OwnedDataValue::from_i64(2)),
        ]);
        let b = OwnedDataValue::Object(vec![
            ("y".to_string(), OwnedDataValue::from_i64(2)),
            ("x".to_string(), OwnedDataValue::from_i64(1)),
        ]);
        assert_eq!(a, b);
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn datetime_variant_round_trips_through_owned() {
        use crate::datetime::DataDateTime;
        let arena = Bump::new();
        let dt = DataDateTime::parse("2024-01-15T12:30:45Z").unwrap();
        let bv = DataValue::DateTime(dt);
        let owned = bv.to_owned();
        assert!(owned.is_datetime());
        assert_eq!(
            owned.as_datetime().unwrap().to_iso_string(),
            "2024-01-15T12:30:45Z"
        );
        let back = owned.to_arena(&arena);
        assert_eq!(back, bv);
        assert_eq!(owned.view_in(&arena), bv);
    }

    #[cfg(feature = "tensor")]
    #[test]
    fn view_in_borrows_tensor_shape_and_bytes() {
        use crate::tensor::{DataTensor, OwnedDataTensor};
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let t = OwnedDataTensor::from_slice([4, 256], &data).unwrap();
        let owned = OwnedDataValue::tensor(t);
        let src = owned.as_tensor().unwrap();

        let arena = Bump::new();
        let view = owned.view_in(&arena);
        let vt = view.as_tensor().unwrap();
        assert_eq!(vt.data().as_ptr(), src.data().as_ptr());
        assert_eq!(vt.shape().as_ptr(), src.shape().as_ptr());
        assert_eq!(vt.as_slice::<f32>(), Some(&data[..]));
        assert_eq!(view, owned.to_arena(&Bump::new()));

        // The 4 KiB payload stays put; only the header is allocated.
        let used = arena_bytes_used(|a| {
            owned.view_in(a);
        });
        assert!(used <= size_of::<DataTensor<'_>>() + 8, "used {used}");
    }

    #[cfg(feature = "tensor")]
    #[test]
    fn tensor_variant_round_trips_and_shares_on_clone() {
        use crate::tensor::{DataTensor, OwnedDataTensor};
        let arena = Bump::new();
        let t = DataTensor::from_slice_in(&[3], &[1i64, 2, 3], &arena).unwrap();
        let bv = DataValue::tensor_in(t, &arena);
        let owned = bv.to_owned();
        assert!(owned.is_tensor());
        assert_eq!(
            owned.as_tensor().unwrap().as_slice::<i64>(),
            Some(&[1i64, 2, 3][..])
        );
        assert!(owned.as_array().is_none());
        assert!(owned["x"].is_null());
        assert_eq!(owned.len(), None);
        drop(arena);

        let cloned = owned.clone();
        match (&owned, &cloned) {
            (OwnedDataValue::Tensor(a), OwnedDataValue::Tensor(b)) => {
                assert!(Arc::ptr_eq(a, b), "clone shares the buffer");
            }
            _ => unreachable!(),
        }
        assert_eq!(owned, cloned);

        let arena2 = Bump::new();
        let back = owned.to_arena(&arena2);
        assert_eq!(back.to_owned(), owned);
        assert_eq!(back.as_tensor().unwrap().shape(), &[3]);

        let direct = OwnedDataTensor::from_slice([3], &[1i64, 2, 3]).unwrap();
        let v = crate::owned_json!({"t": direct, "n": 1});
        assert!(v["t"].is_tensor());
        assert_eq!(v["t"], owned);
    }
}
