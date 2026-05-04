//! [`DataValue`] — bump-allocated JSON value type.
//!
//! Lifetime `'a` ties the value tree to a [`bumpalo::Bump`]. Composite
//! variants (`String`, `Array`, `Object`) hold arena-allocated slices, so
//! constructing a `DataValue` tree costs one or two arena bumps per node
//! instead of a heap allocation per `Vec` / `BTreeMap` / `String`.

use core::ops::Index;

use bumpalo::Bump;

#[cfg(feature = "datetime")]
use crate::datetime::{DataDateTime, DataDuration};
use crate::number::NumberValue;

/// Arena-allocated JSON value tree. Mirrors `serde_json::Value` in shape
/// and access surface, but every composite payload lives in a `Bump`.
#[derive(Debug, Clone, Copy)]
pub enum DataValue<'a> {
    Null,
    Bool(bool),
    Number(NumberValue),
    String(&'a str),
    Array(&'a [DataValue<'a>]),
    Object(&'a [(&'a str, DataValue<'a>)]),
    /// UTC instant + original tz offset. JSON has no native datetime, so
    /// the JSON parser never produces this — consumers upgrade from
    /// `String` at the operator boundary.
    #[cfg(feature = "datetime")]
    DateTime(DataDateTime),
    /// Signed duration. Same boundary rules as `DateTime`.
    #[cfg(feature = "datetime")]
    Duration(DataDuration),
}

/// Returned by `Index` impls when a key/index is missing — matches
/// `serde_json::Value`'s "indexing returns Null on miss" behaviour.
pub(crate) static NULL: DataValue<'static> = DataValue::Null;

impl<'a> DataValue<'a> {
    // ---- Constructors ----

    #[inline]
    pub fn null() -> Self {
        DataValue::Null
    }

    #[inline]
    pub fn bool(b: bool) -> Self {
        DataValue::Bool(b)
    }

    #[inline]
    pub fn from_i64(i: i64) -> Self {
        DataValue::Number(NumberValue::from_i64(i))
    }

    #[inline]
    pub fn from_f64(f: f64) -> Self {
        DataValue::Number(NumberValue::from_f64(f))
    }

    #[inline]
    pub fn from_str_in(s: &str, arena: &'a Bump) -> Self {
        DataValue::String(arena.alloc_str(s))
    }

    /// Wrap a string slice that already lives in the arena (or has the
    /// required lifetime). No allocation.
    #[inline]
    pub fn from_borrowed_str(s: &'a str) -> Self {
        DataValue::String(s)
    }

    // ---- Type predicates ----

    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, DataValue::Null)
    }
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(self, DataValue::Bool(_))
    }
    #[inline]
    pub fn is_number(&self) -> bool {
        matches!(self, DataValue::Number(_))
    }
    #[inline]
    pub fn is_i64(&self) -> bool {
        matches!(self, DataValue::Number(NumberValue::Integer(_)))
    }
    #[inline]
    pub fn is_f64(&self) -> bool {
        matches!(self, DataValue::Number(NumberValue::Float(_)))
    }
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, DataValue::String(_))
    }
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, DataValue::Array(_))
    }
    #[inline]
    pub fn is_object(&self) -> bool {
        matches!(self, DataValue::Object(_))
    }

    #[cfg(feature = "datetime")]
    #[inline]
    pub fn is_datetime(&self) -> bool {
        matches!(self, DataValue::DateTime(_))
    }
    #[cfg(feature = "datetime")]
    #[inline]
    pub fn is_duration(&self) -> bool {
        matches!(self, DataValue::Duration(_))
    }

    // ---- Accessors ----

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            DataValue::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            DataValue::Number(n) => Some(n.as_f64()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_number(&self) -> Option<&NumberValue> {
        match self {
            DataValue::Number(n) => Some(n),
            _ => None,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            DataValue::String(s) => Some(s),
            _ => None,
        }
    }

    #[inline]
    pub fn as_array(&self) -> Option<&'a [DataValue<'a>]> {
        match *self {
            DataValue::Array(a) => Some(a),
            _ => None,
        }
    }

    #[inline]
    pub fn as_object(&self) -> Option<&'a [(&'a str, DataValue<'a>)]> {
        match *self {
            DataValue::Object(o) => Some(o),
            _ => None,
        }
    }

    #[cfg(feature = "datetime")]
    #[inline]
    pub fn as_datetime(&self) -> Option<&DataDateTime> {
        match self {
            DataValue::DateTime(d) => Some(d),
            _ => None,
        }
    }

    #[cfg(feature = "datetime")]
    #[inline]
    pub fn as_duration(&self) -> Option<&DataDuration> {
        match self {
            DataValue::Duration(d) => Some(d),
            _ => None,
        }
    }

    #[cfg(feature = "datetime")]
    #[inline]
    pub fn datetime(dt: DataDateTime) -> Self {
        DataValue::DateTime(dt)
    }

    #[cfg(feature = "datetime")]
    #[inline]
    pub fn duration(d: DataDuration) -> Self {
        DataValue::Duration(d)
    }

    /// `serde_json::Value::get`-style lookup. Accepts `&str` for object
    /// keys or `usize` for array indices.
    #[inline]
    pub fn get<I: ValueIndex>(&self, index: I) -> Option<&DataValue<'a>> {
        I::index_into(&index, self)
    }

    /// Number of elements in an array / object. `None` for non-collections.
    #[inline]
    pub fn len(&self) -> Option<usize> {
        match self {
            DataValue::Array(a) => Some(a.len()),
            DataValue::Object(o) => Some(o.len()),
            _ => None,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|n| n == 0)
    }

    /// Iterate array items. Returns an empty iterator if `self` is not an
    /// array — same convenience pattern as `json-rust`'s `members`.
    #[inline]
    pub fn members(&self) -> core::slice::Iter<'_, DataValue<'a>> {
        match *self {
            DataValue::Array(items) => items.iter(),
            _ => [].iter(),
        }
    }

    /// Iterate object entries as `(key, value)` pairs in insertion order.
    /// Returns an empty iterator if `self` is not an object.
    #[inline]
    pub fn entries(&self) -> EntriesIter<'_, 'a> {
        match *self {
            DataValue::Object(pairs) => EntriesIter {
                inner: pairs.iter(),
            },
            _ => EntriesIter { inner: [].iter() },
        }
    }
}

/// Iterator over `(key, value)` pairs in a [`DataValue::Object`]. Created
/// via [`DataValue::entries`].
pub struct EntriesIter<'v, 'a> {
    inner: core::slice::Iter<'v, (&'a str, DataValue<'a>)>,
}

impl<'v, 'a> Iterator for EntriesIter<'v, 'a> {
    type Item = (&'a str, &'v DataValue<'a>);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (*k, v))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for EntriesIter<'_, '_> {}

impl Default for DataValue<'_> {
    #[inline]
    fn default() -> Self {
        DataValue::Null
    }
}

impl<'a> PartialEq for DataValue<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DataValue::Null, DataValue::Null) => true,
            (DataValue::Bool(a), DataValue::Bool(b)) => a == b,
            (DataValue::Number(a), DataValue::Number(b)) => a == b,
            (DataValue::String(a), DataValue::String(b)) => a == b,
            (DataValue::Array(a), DataValue::Array(b)) => a == b,
            (DataValue::Object(a), DataValue::Object(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                // Object equality is by key set, not key order — match serde_json.
                a.iter().all(|(k, v)| {
                    b.iter()
                        .find(|(bk, _)| bk == k)
                        .is_some_and(|(_, bv)| v == bv)
                })
            }
            #[cfg(feature = "datetime")]
            (DataValue::DateTime(a), DataValue::DateTime(b)) => a == b,
            #[cfg(feature = "datetime")]
            (DataValue::Duration(a), DataValue::Duration(b)) => a == b,
            _ => false,
        }
    }
}

// ---- Index trait dispatch ----

/// Sealed-style helper for `DataValue::get`. Implemented for `&str`,
/// `String`, and `usize`.
pub trait ValueIndex: private::Sealed {
    fn index_into<'v, 'a>(&self, value: &'v DataValue<'a>) -> Option<&'v DataValue<'a>>;
    fn index_into_or_null<'v, 'a>(&self, value: &'v DataValue<'a>) -> &'v DataValue<'a>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for str {}
    impl Sealed for String {}
    impl Sealed for usize {}
    impl<T: Sealed + ?Sized> Sealed for &T {}
}

impl ValueIndex for str {
    #[inline]
    fn index_into<'v, 'a>(&self, value: &'v DataValue<'a>) -> Option<&'v DataValue<'a>> {
        match value {
            DataValue::Object(pairs) => pairs.iter().find(|(k, _)| *k == self).map(|(_, v)| v),
            _ => None,
        }
    }
    #[inline]
    fn index_into_or_null<'v, 'a>(&self, value: &'v DataValue<'a>) -> &'v DataValue<'a> {
        // &NULL is &'static DataValue<'static>; covariance in 'a coerces it
        // to &'v DataValue<'a> since 'static: 'a.
        self.index_into(value).unwrap_or(&NULL)
    }
}

impl ValueIndex for String {
    #[inline]
    fn index_into<'v, 'a>(&self, value: &'v DataValue<'a>) -> Option<&'v DataValue<'a>> {
        self.as_str().index_into(value)
    }
    #[inline]
    fn index_into_or_null<'v, 'a>(&self, value: &'v DataValue<'a>) -> &'v DataValue<'a> {
        self.as_str().index_into_or_null(value)
    }
}

impl ValueIndex for usize {
    #[inline]
    fn index_into<'v, 'a>(&self, value: &'v DataValue<'a>) -> Option<&'v DataValue<'a>> {
        match value {
            DataValue::Array(items) => items.get(*self),
            _ => None,
        }
    }
    #[inline]
    fn index_into_or_null<'v, 'a>(&self, value: &'v DataValue<'a>) -> &'v DataValue<'a> {
        self.index_into(value).unwrap_or(&NULL)
    }
}

impl<T: ValueIndex + ?Sized> ValueIndex for &T {
    #[inline]
    fn index_into<'v, 'a>(&self, value: &'v DataValue<'a>) -> Option<&'v DataValue<'a>> {
        (**self).index_into(value)
    }
    #[inline]
    fn index_into_or_null<'v, 'a>(&self, value: &'v DataValue<'a>) -> &'v DataValue<'a> {
        (**self).index_into_or_null(value)
    }
}

impl<'a, I: ValueIndex> Index<I> for DataValue<'a> {
    type Output = DataValue<'a>;
    #[inline]
    fn index(&self, index: I) -> &DataValue<'a> {
        index.index_into_or_null(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<'a>(arena: &'a Bump) -> DataValue<'a> {
        let nested = arena.alloc_slice_copy(&[
            DataValue::from_i64(1),
            DataValue::from_i64(2),
            DataValue::from_i64(3),
        ]);
        let inner_obj = arena.alloc_slice_copy(&[("k", DataValue::Bool(true))]);
        let pairs = arena.alloc_slice_copy(&[
            ("name", DataValue::from_borrowed_str("alice")),
            ("nums", DataValue::Array(nested)),
            ("inner", DataValue::Object(inner_obj)),
        ]);
        DataValue::Object(pairs)
    }

    #[test]
    fn get_object_key() {
        let arena = Bump::new();
        let v = sample(&arena);
        assert_eq!(v.get("name").and_then(|x| x.as_str()), Some("alice"));
        assert!(v.get("missing").is_none());
    }

    #[test]
    fn get_array_index() {
        let arena = Bump::new();
        let v = sample(&arena);
        let nums = v.get("nums").unwrap();
        assert_eq!(nums.get(0).and_then(|x| x.as_i64()), Some(1));
        assert_eq!(nums.get(2).and_then(|x| x.as_i64()), Some(3));
        assert!(nums.get(99).is_none());
    }

    #[test]
    fn index_returns_null_for_missing() {
        let arena = Bump::new();
        let v = sample(&arena);
        assert!(v["missing"].is_null());
        assert!(v["nums"][99].is_null());
    }

    #[test]
    fn chained_index() {
        let arena = Bump::new();
        let v = sample(&arena);
        assert_eq!(v["inner"]["k"].as_bool(), Some(true));
    }

    #[test]
    fn predicates_and_len() {
        let arena = Bump::new();
        let v = sample(&arena);
        assert!(v.is_object());
        assert_eq!(v.len(), Some(3));
        assert_eq!(v["nums"].len(), Some(3));
        assert_eq!(v["name"].len(), None);
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn datetime_variant_round_trips_through_value() {
        use crate::datetime::{DataDateTime, DataDuration};
        let dt = DataDateTime::parse("2024-01-15T12:30:45Z").unwrap();
        let v = DataValue::datetime(dt);
        assert!(v.is_datetime());
        assert_eq!(
            v.as_datetime().map(|d| d.to_iso_string()).as_deref(),
            Some("2024-01-15T12:30:45Z")
        );

        let dur = DataDuration::parse("1d:2h").unwrap();
        let v2 = DataValue::duration(dur);
        assert!(v2.is_duration());
        assert_eq!(v2.as_duration().unwrap().to_string(), "1d:2h:0m:0s");

        // Equality on the parent enum dispatches to the variant.
        assert_eq!(v, DataValue::datetime(dt));
        assert_ne!(v, v2);
    }

    #[test]
    fn equality_object_order_insensitive() {
        let arena = Bump::new();
        let a =
            arena.alloc_slice_copy(&[("x", DataValue::from_i64(1)), ("y", DataValue::from_i64(2))]);
        let b =
            arena.alloc_slice_copy(&[("y", DataValue::from_i64(2)), ("x", DataValue::from_i64(1))]);
        assert_eq!(DataValue::Object(a), DataValue::Object(b));
    }
}
