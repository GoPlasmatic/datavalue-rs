//! Serde integration. Gated behind the `serde` feature.
//!
//! - [`Serialize`] for [`DataValue`] — straightforward, no lifetime drama.
//! - [`DataValueSeed`] — a [`DeserializeSeed`] adapter that carries the
//!   `&Bump` so [`DataValue`] can be reconstructed from any serde data
//!   format.
//!
//! Note: [`DataValue::from_str`] (the hand-rolled parser) is the fast path
//! for JSON input. The seed exists so callers can plug into existing serde
//! pipelines (e.g. flexbuffers, msgpack, or `serde_json::Deserializer`).

use core::fmt;
use core::marker::PhantomData;

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use serde::de::{DeserializeSeed, Deserializer, Error as DeError, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use crate::number::NumberValue;
use crate::owned::OwnedDataValue;
use crate::value::DataValue;

impl Serialize for DataValue<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match *self {
            DataValue::Null => serializer.serialize_unit(),
            DataValue::Bool(b) => serializer.serialize_bool(b),
            DataValue::Number(n) => match n {
                NumberValue::Integer(i) => serializer.serialize_i64(i),
                NumberValue::Float(f) => serializer.serialize_f64(f),
            },
            DataValue::String(s) => serializer.serialize_str(s),
            DataValue::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            DataValue::Object(pairs) => {
                let mut map = serializer.serialize_map(Some(pairs.len()))?;
                for (k, v) in pairs {
                    map.serialize_entry(*k, v)?;
                }
                map.end()
            }
            // JSON has no datetime/duration types — render as strings using
            // the same wire format the parser side accepts.
            #[cfg(feature = "datetime")]
            DataValue::DateTime(d) => serializer.serialize_str(&d.to_iso_string()),
            #[cfg(feature = "datetime")]
            DataValue::Duration(d) => serializer.collect_str(&d),
        }
    }
}

/// Deserialize a [`DataValue`] tree into a borrowed [`Bump`] arena.
///
/// ```ignore
/// use bumpalo::Bump;
/// use datavalue_rs::DataValueSeed;
/// use serde::de::DeserializeSeed;
///
/// let arena = Bump::new();
/// let mut de = serde_json::Deserializer::from_str(r#"{"x":1}"#);
/// let v = DataValueSeed::new(&arena).deserialize(&mut de).unwrap();
/// assert_eq!(v["x"].as_i64(), Some(1));
/// ```
#[derive(Clone, Copy)]
pub struct DataValueSeed<'a> {
    arena: &'a Bump,
}

impl<'a> DataValueSeed<'a> {
    #[inline]
    pub fn new(arena: &'a Bump) -> Self {
        Self { arena }
    }
}

impl<'de, 'a> DeserializeSeed<'de> for DataValueSeed<'a> {
    type Value = DataValue<'a>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(DataValueVisitor {
            arena: self.arena,
            _de: PhantomData,
        })
    }
}

struct DataValueVisitor<'a, 'de> {
    arena: &'a Bump,
    _de: PhantomData<&'de ()>,
}

impl<'a, 'de> Visitor<'de> for DataValueVisitor<'a, 'de> {
    type Value = DataValue<'a>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any valid JSON value")
    }

    #[inline]
    fn visit_unit<E: DeError>(self) -> Result<Self::Value, E> {
        Ok(DataValue::Null)
    }
    #[inline]
    fn visit_none<E: DeError>(self) -> Result<Self::Value, E> {
        Ok(DataValue::Null)
    }
    #[inline]
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }

    #[inline]
    fn visit_bool<E: DeError>(self, v: bool) -> Result<Self::Value, E> {
        Ok(DataValue::Bool(v))
    }

    #[inline]
    fn visit_i64<E: DeError>(self, v: i64) -> Result<Self::Value, E> {
        Ok(DataValue::Number(NumberValue::Integer(v)))
    }
    #[inline]
    fn visit_i128<E: DeError>(self, v: i128) -> Result<Self::Value, E> {
        if (i64::MIN as i128..=i64::MAX as i128).contains(&v) {
            Ok(DataValue::Number(NumberValue::Integer(v as i64)))
        } else {
            Ok(DataValue::Number(NumberValue::Float(v as f64)))
        }
    }
    #[inline]
    fn visit_u64<E: DeError>(self, v: u64) -> Result<Self::Value, E> {
        if v <= i64::MAX as u64 {
            Ok(DataValue::Number(NumberValue::Integer(v as i64)))
        } else {
            Ok(DataValue::Number(NumberValue::Float(v as f64)))
        }
    }
    #[inline]
    fn visit_u128<E: DeError>(self, v: u128) -> Result<Self::Value, E> {
        if v <= i64::MAX as u128 {
            Ok(DataValue::Number(NumberValue::Integer(v as i64)))
        } else {
            Ok(DataValue::Number(NumberValue::Float(v as f64)))
        }
    }
    #[inline]
    fn visit_f64<E: DeError>(self, v: f64) -> Result<Self::Value, E> {
        Ok(DataValue::Number(NumberValue::from_f64(v)))
    }

    #[inline]
    fn visit_str<E: DeError>(self, v: &str) -> Result<Self::Value, E> {
        Ok(DataValue::String(self.arena.alloc_str(v)))
    }
    #[inline]
    fn visit_borrowed_str<E: DeError>(self, v: &'de str) -> Result<Self::Value, E> {
        // We cannot statically claim 'de outlives 'a, so copy into the arena.
        Ok(DataValue::String(self.arena.alloc_str(v)))
    }
    #[inline]
    fn visit_string<E: DeError>(self, v: String) -> Result<Self::Value, E> {
        Ok(DataValue::String(self.arena.alloc_str(&v)))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let cap = seq.size_hint().unwrap_or(0);
        let mut items: BumpVec<DataValue<'a>> = BumpVec::with_capacity_in(cap, self.arena);
        while let Some(v) = seq.next_element_seed(DataValueSeed { arena: self.arena })? {
            items.push(v);
        }
        Ok(DataValue::Array(items.into_bump_slice()))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let cap = map.size_hint().unwrap_or(0);
        let mut pairs: BumpVec<(&'a str, DataValue<'a>)> =
            BumpVec::with_capacity_in(cap, self.arena);
        // Keys come through as Strings — copy into arena.
        while let Some(k) = map.next_key::<String>()? {
            let v = map.next_value_seed(DataValueSeed { arena: self.arena })?;
            pairs.push((self.arena.alloc_str(&k), v));
        }
        Ok(DataValue::Object(pairs.into_bump_slice()))
    }
}

// ---- OwnedDataValue: full serde Serialize / Deserialize, no seed needed. ----

impl Serialize for OwnedDataValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            OwnedDataValue::Null => serializer.serialize_unit(),
            OwnedDataValue::Bool(b) => serializer.serialize_bool(*b),
            OwnedDataValue::Number(n) => match *n {
                NumberValue::Integer(i) => serializer.serialize_i64(i),
                NumberValue::Float(f) => serializer.serialize_f64(f),
            },
            OwnedDataValue::String(s) => serializer.serialize_str(s),
            OwnedDataValue::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            OwnedDataValue::Object(pairs) => {
                let mut map = serializer.serialize_map(Some(pairs.len()))?;
                for (k, v) in pairs {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
            #[cfg(feature = "datetime")]
            OwnedDataValue::DateTime(d) => serializer.serialize_str(&d.to_iso_string()),
            #[cfg(feature = "datetime")]
            OwnedDataValue::Duration(d) => serializer.collect_str(d),
        }
    }
}

impl<'de> serde::Deserialize<'de> for OwnedDataValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OwnedDataValueVisitor)
    }
}

struct OwnedDataValueVisitor;

impl<'de> Visitor<'de> for OwnedDataValueVisitor {
    type Value = OwnedDataValue;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any valid JSON value")
    }

    #[inline]
    fn visit_unit<E: DeError>(self) -> Result<Self::Value, E> {
        Ok(OwnedDataValue::Null)
    }
    #[inline]
    fn visit_none<E: DeError>(self) -> Result<Self::Value, E> {
        Ok(OwnedDataValue::Null)
    }
    #[inline]
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }
    #[inline]
    fn visit_bool<E: DeError>(self, v: bool) -> Result<Self::Value, E> {
        Ok(OwnedDataValue::Bool(v))
    }
    #[inline]
    fn visit_i64<E: DeError>(self, v: i64) -> Result<Self::Value, E> {
        Ok(OwnedDataValue::Number(NumberValue::Integer(v)))
    }
    #[inline]
    fn visit_i128<E: DeError>(self, v: i128) -> Result<Self::Value, E> {
        if (i64::MIN as i128..=i64::MAX as i128).contains(&v) {
            Ok(OwnedDataValue::Number(NumberValue::Integer(v as i64)))
        } else {
            Ok(OwnedDataValue::Number(NumberValue::Float(v as f64)))
        }
    }
    #[inline]
    fn visit_u64<E: DeError>(self, v: u64) -> Result<Self::Value, E> {
        if v <= i64::MAX as u64 {
            Ok(OwnedDataValue::Number(NumberValue::Integer(v as i64)))
        } else {
            Ok(OwnedDataValue::Number(NumberValue::Float(v as f64)))
        }
    }
    #[inline]
    fn visit_u128<E: DeError>(self, v: u128) -> Result<Self::Value, E> {
        if v <= i64::MAX as u128 {
            Ok(OwnedDataValue::Number(NumberValue::Integer(v as i64)))
        } else {
            Ok(OwnedDataValue::Number(NumberValue::Float(v as f64)))
        }
    }
    #[inline]
    fn visit_f64<E: DeError>(self, v: f64) -> Result<Self::Value, E> {
        Ok(OwnedDataValue::Number(NumberValue::from_f64(v)))
    }
    #[inline]
    fn visit_str<E: DeError>(self, v: &str) -> Result<Self::Value, E> {
        Ok(OwnedDataValue::String(v.to_string()))
    }
    #[inline]
    fn visit_string<E: DeError>(self, v: String) -> Result<Self::Value, E> {
        Ok(OwnedDataValue::String(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(v) = seq.next_element::<OwnedDataValue>()? {
            items.push(v);
        }
        Ok(OwnedDataValue::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut pairs = Vec::with_capacity(map.size_hint().unwrap_or(0));
        while let Some(k) = map.next_key::<String>()? {
            let v = map.next_value::<OwnedDataValue>()?;
            pairs.push((k, v));
        }
        Ok(OwnedDataValue::Object(pairs))
    }
}

#[cfg(all(test, feature = "serde_json"))]
mod tests {
    use super::*;
    use serde::de::DeserializeSeed;
    use serde_json::json;

    #[test]
    fn round_trip_via_serde_json() {
        let arena = Bump::new();
        let input = json!({
            "name": "alice",
            "age": 30,
            "scores": [1, 2.5, null],
            "active": true,
        });
        let s = serde_json::to_string(&input).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let v = DataValueSeed::new(&arena).deserialize(&mut de).unwrap();
        assert_eq!(v["name"].as_str(), Some("alice"));
        assert_eq!(v["age"].as_i64(), Some(30));
        assert_eq!(v["scores"][0].as_i64(), Some(1));
        assert_eq!(v["scores"][1].as_f64(), Some(2.5));
        assert!(v["scores"][2].is_null());
        assert_eq!(v["active"].as_bool(), Some(true));

        let back = serde_json::to_value(v).unwrap();
        assert_eq!(back, input);
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn datetime_serializes_as_iso_string() {
        use crate::datetime::{DataDateTime, DataDuration};
        let dt = DataDateTime::parse("2024-01-15T12:30:45Z").unwrap();
        let dur = DataDuration::parse("3d:4h").unwrap();
        let v = DataValue::DateTime(dt);
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#""2024-01-15T12:30:45Z""#
        );
        let v = DataValue::Duration(dur);
        assert_eq!(serde_json::to_string(&v).unwrap(), r#""3d:4h:0m:0s""#);
    }

    #[test]
    fn serialize_matches_input() {
        let arena = Bump::new();
        let input = r#"{"a":1,"b":"x","c":[true,null,1.5]}"#;
        let v = DataValue::from_str(input, &arena).unwrap();
        let s = serde_json::to_string(&v).unwrap();
        // Object key order is preserved in the parser, so this should match exactly.
        assert_eq!(s, input);
    }

    #[test]
    fn owned_round_trips_via_serde_json() {
        let input = r#"{"name":"alice","ages":[30,31],"active":true}"#;
        let v: OwnedDataValue = serde_json::from_str(input).unwrap();
        assert_eq!(v["name"].as_str(), Some("alice"));
        assert_eq!(v["ages"][1].as_i64(), Some(31));
        assert_eq!(v["active"].as_bool(), Some(true));

        let back = serde_json::to_string(&v).unwrap();
        assert_eq!(back, input);
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn owned_datetime_serializes_as_string() {
        use crate::datetime::DataDateTime;
        let dt = DataDateTime::parse("2024-01-15T12:30:45Z").unwrap();
        let v = OwnedDataValue::DateTime(dt);
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#""2024-01-15T12:30:45Z""#
        );
    }
}
