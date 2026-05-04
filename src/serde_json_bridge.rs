//! Bidirectional bridge to [`serde_json::Value`].
//!
//! Walks the source tree directly — no round-trip through a JSON string —
//! so this is faster than `serde_json::to_string` + `from_str`. Use it at
//! API boundaries that already speak `serde_json::Value` (e.g. configuration
//! files parsed by upstream code, Axum/Actix request bodies, integrations
//! with crates whose public types are `serde_json::Value`-shaped).
//!
//! Gated behind the `serde_json` feature.

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use serde_json::Value as SjValue;

use crate::number::NumberValue;
use crate::owned::OwnedDataValue;
use crate::value::DataValue;

// ---- OwnedDataValue <-> serde_json::Value -----------------------------------

impl From<&SjValue> for OwnedDataValue {
    fn from(v: &SjValue) -> Self {
        match v {
            SjValue::Null => OwnedDataValue::Null,
            SjValue::Bool(b) => OwnedDataValue::Bool(*b),
            SjValue::Number(n) => OwnedDataValue::Number(sj_number_to_native(n)),
            SjValue::String(s) => OwnedDataValue::String(s.clone()),
            SjValue::Array(items) => {
                OwnedDataValue::Array(items.iter().map(OwnedDataValue::from).collect())
            }
            SjValue::Object(map) => OwnedDataValue::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), OwnedDataValue::from(v)))
                    .collect(),
            ),
        }
    }
}

impl From<SjValue> for OwnedDataValue {
    /// Move-out variant: avoids cloning string payloads that are already on
    /// the heap inside `serde_json::Value`.
    fn from(v: SjValue) -> Self {
        match v {
            SjValue::Null => OwnedDataValue::Null,
            SjValue::Bool(b) => OwnedDataValue::Bool(b),
            SjValue::Number(n) => OwnedDataValue::Number(sj_number_to_native(&n)),
            SjValue::String(s) => OwnedDataValue::String(s),
            SjValue::Array(items) => {
                OwnedDataValue::Array(items.into_iter().map(OwnedDataValue::from).collect())
            }
            SjValue::Object(map) => OwnedDataValue::Object(
                map.into_iter()
                    .map(|(k, v)| (k, OwnedDataValue::from(v)))
                    .collect(),
            ),
        }
    }
}

impl From<&OwnedDataValue> for SjValue {
    fn from(v: &OwnedDataValue) -> Self {
        match v {
            OwnedDataValue::Null => SjValue::Null,
            OwnedDataValue::Bool(b) => SjValue::Bool(*b),
            OwnedDataValue::Number(n) => native_number_to_sj(*n),
            OwnedDataValue::String(s) => SjValue::String(s.clone()),
            OwnedDataValue::Array(items) => {
                SjValue::Array(items.iter().map(SjValue::from).collect())
            }
            OwnedDataValue::Object(pairs) => {
                let mut map = serde_json::Map::with_capacity(pairs.len());
                for (k, v) in pairs {
                    map.insert(k.clone(), SjValue::from(v));
                }
                SjValue::Object(map)
            }
            #[cfg(feature = "datetime")]
            OwnedDataValue::DateTime(d) => SjValue::String(d.to_iso_string()),
            #[cfg(feature = "datetime")]
            OwnedDataValue::Duration(d) => SjValue::String(d.to_string()),
        }
    }
}

impl OwnedDataValue {
    /// Convert from a `serde_json::Value` reference. Equivalent to
    /// `OwnedDataValue::from(value)`; provided as a named entry point
    /// for discoverability.
    #[inline]
    pub fn from_serde_value(value: &SjValue) -> Self {
        Self::from(value)
    }

    /// Convert into a `serde_json::Value`. Equivalent to
    /// `serde_json::Value::from(&value)`.
    #[inline]
    pub fn to_serde_value(&self) -> SjValue {
        SjValue::from(self)
    }
}

// ---- DataValue<'a> <-> serde_json::Value -------------------------------------

impl<'a> DataValue<'a> {
    /// Walk a `serde_json::Value` into a [`DataValue`] tree backed by
    /// `arena`. Strings and object keys are arena-allocated copies.
    pub fn from_serde_value_in(value: &SjValue, arena: &'a Bump) -> Self {
        match value {
            SjValue::Null => DataValue::Null,
            SjValue::Bool(b) => DataValue::Bool(*b),
            SjValue::Number(n) => DataValue::Number(sj_number_to_native(n)),
            SjValue::String(s) => DataValue::String(arena.alloc_str(s)),
            SjValue::Array(items) => {
                let mut buf = BumpVec::with_capacity_in(items.len(), arena);
                for item in items {
                    buf.push(DataValue::from_serde_value_in(item, arena));
                }
                DataValue::Array(buf.into_bump_slice())
            }
            SjValue::Object(map) => {
                let mut buf = BumpVec::with_capacity_in(map.len(), arena);
                for (k, v) in map {
                    buf.push((
                        arena.alloc_str(k) as &str,
                        DataValue::from_serde_value_in(v, arena),
                    ));
                }
                DataValue::Object(buf.into_bump_slice())
            }
        }
    }

    /// Materialise this arena-bound tree as a `serde_json::Value` (heap-owned).
    pub fn to_serde_value(&self) -> SjValue {
        match *self {
            DataValue::Null => SjValue::Null,
            DataValue::Bool(b) => SjValue::Bool(b),
            DataValue::Number(n) => native_number_to_sj(n),
            DataValue::String(s) => SjValue::String(s.to_string()),
            DataValue::Array(items) => {
                SjValue::Array(items.iter().map(DataValue::to_serde_value).collect())
            }
            DataValue::Object(pairs) => {
                let mut map = serde_json::Map::with_capacity(pairs.len());
                for (k, v) in pairs {
                    map.insert((*k).to_string(), v.to_serde_value());
                }
                SjValue::Object(map)
            }
            #[cfg(feature = "datetime")]
            DataValue::DateTime(d) => SjValue::String(d.to_iso_string()),
            #[cfg(feature = "datetime")]
            DataValue::Duration(d) => SjValue::String(d.to_string()),
        }
    }
}

// ---- Number conversion -------------------------------------------------------

#[inline]
fn sj_number_to_native(n: &serde_json::Number) -> NumberValue {
    if let Some(i) = n.as_i64() {
        NumberValue::Integer(i)
    } else if let Some(u) = n.as_u64() {
        // u64 above i64::MAX falls through to f64 (matches the parser's u64 path).
        NumberValue::Float(u as f64)
    } else {
        // f64 path. `as_f64` is total on serde_json::Number unless the input
        // is a non-finite literal (which serde_json doesn't accept anyway).
        NumberValue::Float(n.as_f64().unwrap_or(0.0))
    }
}

#[inline]
fn native_number_to_sj(n: NumberValue) -> SjValue {
    match n {
        NumberValue::Integer(i) => SjValue::Number(i.into()),
        NumberValue::Float(f) => {
            // serde_json::Number rejects non-finite floats; emit Null to match
            // our compact-emit behaviour.
            serde_json::Number::from_f64(f).map_or(SjValue::Null, SjValue::Number)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn owned_round_trip_via_serde_json_value() {
        let sj = json!({
            "name": "alice",
            "age": 30,
            "scores": [1, 2.5, null],
            "active": true,
            "nested": {"flag": false, "list": [1, 2, 3]},
        });
        let owned: OwnedDataValue = OwnedDataValue::from_serde_value(&sj);
        assert_eq!(owned["name"].as_str(), Some("alice"));
        assert_eq!(owned["age"].as_i64(), Some(30));
        assert_eq!(owned["scores"][0].as_i64(), Some(1));
        assert!((owned["scores"][1].as_f64().unwrap() - 2.5).abs() < 1e-9);
        assert!(owned["scores"][2].is_null());
        assert_eq!(owned["active"].as_bool(), Some(true));
        assert_eq!(owned["nested"]["list"][2].as_i64(), Some(3));

        // Round-trip back.
        let back: SjValue = owned.to_serde_value();
        assert_eq!(back, sj);
    }

    #[test]
    fn arena_round_trip_via_serde_json_value() {
        let sj = json!({"x": 42, "y": ["a", "b"], "z": {"k": null}});
        let arena = Bump::new();
        let v = DataValue::from_serde_value_in(&sj, &arena);
        assert_eq!(v["x"].as_i64(), Some(42));
        assert_eq!(v["y"][1].as_str(), Some("b"));
        assert!(v["z"]["k"].is_null());

        let back = v.to_serde_value();
        assert_eq!(back, sj);
    }

    #[test]
    fn move_form_avoids_extra_clone_compiles() {
        // Just verify that `From<SjValue>` (move) typechecks and produces the
        // expected variant. Real allocator impact isn't testable here.
        let sj = json!("hello");
        let owned: OwnedDataValue = sj.into();
        assert_eq!(owned.as_str(), Some("hello"));
    }

    #[test]
    fn u64_above_i64_max_round_trips_as_float() {
        let sj: SjValue = serde_json::from_str("18446744073709551610").unwrap();
        let owned = OwnedDataValue::from_serde_value(&sj);
        assert!(owned.is_f64());
    }

    #[test]
    fn non_finite_float_round_trips_as_null() {
        let v = OwnedDataValue::Number(NumberValue::Float(f64::NAN));
        let sj = v.to_serde_value();
        assert!(sj.is_null());
    }
}
