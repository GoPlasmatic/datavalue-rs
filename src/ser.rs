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
            DataValue::DateTime(d) => serializer.collect_str(&d),
            #[cfg(feature = "datetime")]
            DataValue::Duration(d) => serializer.collect_str(&d),
            #[cfg(feature = "tensor")]
            DataValue::Tensor(t) => t.serialize(serializer),
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

// ---- Tensor: tagged map, `data` base64 in every format. ----

#[cfg(feature = "tensor")]
mod tensor_ser {
    use serde::ser::{Serialize, SerializeMap, Serializer};

    use crate::tensor::{DataTensor, KEY_DATA, KEY_DTYPE, KEY_SHAPE, OwnedDataTensor};

    /// `{"tensor": {"dtype": .., "shape": [..], "data": ..}}`, with `data` a
    /// base64 string in *every* format, binary ones included.
    ///
    /// A byte string would be the natural payload for msgpack or bincode and
    /// 33% smaller, but nothing in this crate can read one back: neither
    /// visitor implements `visit_bytes`, and the honest target for one would
    /// be a `Bytes` variant — a third foreign type, which the crate defers to
    /// an extension slot. Encoding uniformly keeps the documented lossy
    /// round-trip *recoverable* instead: a tensor deserializes as a plain
    /// `Object` whose `data` is base64, and `DataTensor::from_json_value`
    /// rebuilds the tensor from it, in binary formats exactly as in JSON.
    impl Serialize for DataTensor<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(1))?;
            map.serialize_entry(DataTensor::JSON_TAG, &Body(*self))?;
            map.end()
        }
    }

    impl Serialize for OwnedDataTensor {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.view().serialize(serializer)
        }
    }

    struct Body<'a>(DataTensor<'a>);

    impl Serialize for Body<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(3))?;
            map.serialize_entry(KEY_DTYPE, self.0.dtype().name())?;
            map.serialize_entry(KEY_SHAPE, self.0.shape())?;
            map.serialize_entry(KEY_DATA, &Base64Str(self.0.data()))?;
            map.end()
        }
    }

    /// Streams through `collect_str`, so formats that support it (serde_json
    /// does) never materialise the whole base64 string.
    struct Base64Str<'b>(&'b [u8]);

    impl Serialize for Base64Str<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_str(&crate::base64::Base64(self.0))
        }
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
            OwnedDataValue::DateTime(d) => serializer.collect_str(d),
            #[cfg(feature = "datetime")]
            OwnedDataValue::Duration(d) => serializer.collect_str(d),
            // Deref rather than `Arc<T>: Serialize`, which needs serde's `rc` feature.
            #[cfg(feature = "tensor")]
            OwnedDataValue::Tensor(t) => (**t).serialize(serializer),
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

    #[cfg(feature = "tensor")]
    #[test]
    fn tensor_serializes_as_tagged_base64_for_json() {
        use crate::tensor::{DType, DataTensor};
        let arena = Bump::new();
        let t =
            DataTensor::from_slice_in(&[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &arena).unwrap();
        let v = DataValue::tensor_in(t, &arena);
        let expected =
            r#"{"tensor":{"dtype":"f32","shape":[2,3],"data":"AACAPwAAAEAAAEBAAACAQAAAoEAAAMBA"}}"#;
        assert_eq!(serde_json::to_string(&v).unwrap(), expected);
        assert_eq!(serde_json::to_string(&v.to_owned()).unwrap(), expected);
        assert_eq!(serde_json::to_string(&t).unwrap(), expected);
        assert_eq!(serde_json::to_string(&v).unwrap(), v.to_string());

        // Deserialize never produces Tensor: the tagged form is an Object.
        let back: OwnedDataValue = serde_json::from_str(expected).unwrap();
        assert!(back.is_object());
        assert!(back["tensor"]["dtype"].as_str() == Some("f32"));
        let mut de = serde_json::Deserializer::from_str(expected);
        let seeded = DataValueSeed::new(&arena).deserialize(&mut de).unwrap();
        assert!(seeded.is_object());

        let empty = DataTensor::from_bytes_in(DType::F64, &[0], &[], &arena).unwrap();
        assert_eq!(
            serde_json::to_string(&empty).unwrap(),
            r#"{"tensor":{"dtype":"f64","shape":[0],"data":""}}"#
        );
    }

    /// A minimal non-human-readable serializer that records what it is
    /// handed, so what binary formats receive is observable without a
    /// binary-format dev-dependency. `serialize_bytes` records too: if a
    /// raw-bytes branch ever came back, the assertions below would catch it.
    #[cfg(feature = "tensor")]
    mod recording {
        use core::fmt;

        use serde::ser::{self, Impossible, Serialize};

        #[derive(Debug)]
        pub struct Error(String);
        impl fmt::Display for Error {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl std::error::Error for Error {}
        impl ser::Error for Error {
            fn custom<T: fmt::Display>(msg: T) -> Self {
                Error(msg.to_string())
            }
        }

        #[derive(Default)]
        pub struct Rec(pub Vec<String>);

        macro_rules! prim {
            ($($m:ident : $t:ty),* $(,)?) => {$(
                fn $m(self, v: $t) -> Result<(), Error> {
                    self.0.push(format!("{}:{}", stringify!($m), v));
                    Ok(())
                }
            )*};
        }

        impl ser::Serializer for &mut Rec {
            type Ok = ();
            type Error = Error;
            type SerializeSeq = Self;
            type SerializeTuple = Impossible<(), Error>;
            type SerializeTupleStruct = Impossible<(), Error>;
            type SerializeTupleVariant = Impossible<(), Error>;
            type SerializeMap = Self;
            type SerializeStruct = Impossible<(), Error>;
            type SerializeStructVariant = Impossible<(), Error>;

            fn is_human_readable(&self) -> bool {
                false
            }

            prim!(
                serialize_bool: bool, serialize_i8: i8, serialize_i16: i16, serialize_i32: i32,
                serialize_i64: i64, serialize_u8: u8, serialize_u16: u16, serialize_u32: u32,
                serialize_u64: u64, serialize_f32: f32, serialize_f64: f64, serialize_char: char,
                serialize_str: &str,
            );

            fn serialize_bytes(self, v: &[u8]) -> Result<(), Error> {
                self.0.push(format!("bytes:{}", v.len()));
                Ok(())
            }
            fn serialize_none(self) -> Result<(), Error> {
                unimplemented!()
            }
            fn serialize_some<T: ?Sized + Serialize>(self, _: &T) -> Result<(), Error> {
                unimplemented!()
            }
            fn serialize_unit(self) -> Result<(), Error> {
                unimplemented!()
            }
            fn serialize_unit_struct(self, _: &'static str) -> Result<(), Error> {
                unimplemented!()
            }
            fn serialize_unit_variant(
                self,
                _: &'static str,
                _: u32,
                _: &'static str,
            ) -> Result<(), Error> {
                unimplemented!()
            }
            fn serialize_newtype_struct<T: ?Sized + Serialize>(
                self,
                _: &'static str,
                _: &T,
            ) -> Result<(), Error> {
                unimplemented!()
            }
            fn serialize_newtype_variant<T: ?Sized + Serialize>(
                self,
                _: &'static str,
                _: u32,
                _: &'static str,
                _: &T,
            ) -> Result<(), Error> {
                unimplemented!()
            }
            fn serialize_seq(self, len: Option<usize>) -> Result<Self, Error> {
                self.0.push(format!("seq:{}", len.unwrap_or(0)));
                Ok(self)
            }
            fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Error> {
                unimplemented!()
            }
            fn serialize_tuple_struct(
                self,
                _: &'static str,
                _: usize,
            ) -> Result<Self::SerializeTupleStruct, Error> {
                unimplemented!()
            }
            fn serialize_tuple_variant(
                self,
                _: &'static str,
                _: u32,
                _: &'static str,
                _: usize,
            ) -> Result<Self::SerializeTupleVariant, Error> {
                unimplemented!()
            }
            fn serialize_map(self, len: Option<usize>) -> Result<Self, Error> {
                self.0.push(format!("map:{}", len.unwrap_or(0)));
                Ok(self)
            }
            fn serialize_struct(
                self,
                _: &'static str,
                _: usize,
            ) -> Result<Self::SerializeStruct, Error> {
                unimplemented!()
            }
            fn serialize_struct_variant(
                self,
                _: &'static str,
                _: u32,
                _: &'static str,
                _: usize,
            ) -> Result<Self::SerializeStructVariant, Error> {
                unimplemented!()
            }
        }

        impl ser::SerializeSeq for &mut Rec {
            type Ok = ();
            type Error = Error;
            fn serialize_element<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), Error> {
                v.serialize(&mut **self)
            }
            fn end(self) -> Result<(), Error> {
                self.0.push("end".into());
                Ok(())
            }
        }

        impl ser::SerializeMap for &mut Rec {
            type Ok = ();
            type Error = Error;
            fn serialize_key<T: ?Sized + Serialize>(&mut self, k: &T) -> Result<(), Error> {
                k.serialize(&mut **self)
            }
            fn serialize_value<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), Error> {
                v.serialize(&mut **self)
            }
            fn end(self) -> Result<(), Error> {
                self.0.push("end".into());
                Ok(())
            }
        }
    }

    #[cfg(feature = "tensor")]
    #[test]
    fn tensor_data_is_base64_in_binary_formats_too() {
        use crate::tensor::OwnedDataTensor;
        let t = OwnedDataTensor::from_slice([2, 2], &[1i16, 2, 3, 4]).unwrap();
        let v = OwnedDataValue::tensor(t);
        let mut rec = recording::Rec::default();
        v.serialize(&mut rec).unwrap();
        assert_eq!(
            rec.0,
            vec![
                "map:1",
                "serialize_str:tensor",
                "map:3",
                "serialize_str:dtype",
                "serialize_str:i16",
                "serialize_str:shape",
                "seq:2",
                "serialize_u64:2",
                "serialize_u64:2",
                "end",
                "serialize_str:data",
                // Not `bytes:8`: the payload a binary format receives is the
                // same base64 a JSON one does, so the tagged object it
                // deserializes back into can be re-read by
                // `DataTensor::from_json_value`.
                "serialize_str:AQACAAMABAA=",
                "end",
                "end",
            ]
        );
    }
}
