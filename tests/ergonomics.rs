//! Integration tests for the developer-ergonomics surface added on top of the
//! core arena/owned API: `From` impls, the `owned_json!` macro, iteration
//! helpers, `Display` / pretty-print, and the `serde_json::Value` bridge.

use std::collections::{BTreeMap, HashMap};

use datavalue_rs::{OwnedDataValue, owned_json};

#[test]
fn from_primitive_ints_use_integer_path() {
    let v: OwnedDataValue = 42i32.into();
    assert_eq!(v.as_i64(), Some(42));
    assert!(v.is_i64());

    let v: OwnedDataValue = 17u32.into();
    assert_eq!(v.as_i64(), Some(17));
    assert!(v.is_i64());

    // u64 within i64::MAX stays integer
    let v: OwnedDataValue = 1_000_000_000_000u64.into();
    assert_eq!(v.as_i64(), Some(1_000_000_000_000));

    // u64 above i64::MAX falls through to f64
    let v: OwnedDataValue = (i64::MAX as u64 + 10).into();
    assert!(v.is_f64());
}

#[test]
fn from_floats() {
    let v: OwnedDataValue = 1.5f32.into();
    assert!((v.as_f64().unwrap() - 1.5).abs() < 1e-6);

    // f64 with whole value collapses to integer (matches NumberValue::from_f64).
    let v: OwnedDataValue = 3.0f64.into();
    assert!(v.is_i64());
}

#[test]
fn from_unit_is_null() {
    let v: OwnedDataValue = ().into();
    assert!(v.is_null());
}

#[test]
fn from_option() {
    let some: OwnedDataValue = Some(7i32).into();
    assert_eq!(some.as_i64(), Some(7));
    let none: OwnedDataValue = Option::<i32>::None.into();
    assert!(none.is_null());
}

#[test]
fn from_vec_and_array() {
    let v: OwnedDataValue = vec![1i32, 2, 3].into();
    assert_eq!(v.as_array().unwrap().len(), 3);
    assert_eq!(v[0].as_i64(), Some(1));

    let v: OwnedDataValue = [true, false, true].into();
    assert_eq!(v[1].as_bool(), Some(false));

    // mixed via Into-on-elements: works as long as element type is uniform.
    let v: OwnedDataValue = vec!["a".to_string(), "b".to_string()].into();
    assert_eq!(v[1].as_str(), Some("b"));
}

#[test]
fn from_hashmap_and_btreemap() {
    let mut m: HashMap<String, i32> = HashMap::new();
    m.insert("x".into(), 1);
    m.insert("y".into(), 2);
    let v: OwnedDataValue = m.into();
    assert!(v.is_object());
    // order is not specified, but lookup works.
    assert_eq!(v["x"].as_i64(), Some(1));
    assert_eq!(v["y"].as_i64(), Some(2));

    let mut m: BTreeMap<String, &str> = BTreeMap::new();
    m.insert("a".into(), "alpha");
    m.insert("b".into(), "beta");
    let v: OwnedDataValue = m.into();
    assert_eq!(v["a"].as_str(), Some("alpha"));
}

#[test]
fn from_cow_str() {
    use std::borrow::Cow;
    let v: OwnedDataValue = Cow::Borrowed("hi").into();
    assert_eq!(v.as_str(), Some("hi"));
    let v: OwnedDataValue = Cow::<str>::Owned("there".into()).into();
    assert_eq!(v.as_str(), Some("there"));
}

#[test]
fn macro_primitives_and_collections() {
    let v = owned_json!(null);
    assert!(v.is_null());

    let v = owned_json!(true);
    assert_eq!(v.as_bool(), Some(true));

    let v = owned_json!(42);
    assert_eq!(v.as_i64(), Some(42));

    let v = owned_json!("hello");
    assert_eq!(v.as_str(), Some("hello"));

    let v = owned_json!([]);
    assert!(v.is_array());
    assert_eq!(v.len(), Some(0));

    let v = owned_json!({});
    assert!(v.is_object());
}

#[test]
fn macro_nested_object() {
    let v = owned_json!({
        "name": "alice",
        "ages": [30, 31],
        "active": true,
        "tags": null,
        "nested": {
            "inner": [1, 2, 3],
            "flag": false,
        },
    });
    assert_eq!(v["name"].as_str(), Some("alice"));
    assert_eq!(v["ages"][1].as_i64(), Some(31));
    assert_eq!(v["active"].as_bool(), Some(true));
    assert!(v["tags"].is_null());
    assert_eq!(v["nested"]["inner"][2].as_i64(), Some(3));
    assert_eq!(v["nested"]["flag"].as_bool(), Some(false));
}

#[test]
fn macro_with_variable_substitution() {
    let name = "bob";
    let count: i32 = 7;
    let v = owned_json!({
        "name": name,
        "count": count,
        "doubled": (count * 2),
    });
    assert_eq!(v["name"].as_str(), Some("bob"));
    assert_eq!(v["count"].as_i64(), Some(7));
    assert_eq!(v["doubled"].as_i64(), Some(14));
}

#[test]
fn members_iter_array() {
    let v = owned_json!([1, 2, 3, 4]);
    let collected: Vec<i64> = v.members().map(|m| m.as_i64().unwrap()).collect();
    assert_eq!(collected, vec![1, 2, 3, 4]);

    // Non-array yields empty.
    let v = owned_json!("not-an-array");
    assert_eq!(v.members().count(), 0);
}

#[test]
fn entries_iter_object_preserves_order() {
    let v = owned_json!({
        "a": 1,
        "b": 2,
        "c": 3,
    });
    let keys: Vec<&str> = v.entries().map(|(k, _)| k).collect();
    assert_eq!(keys, vec!["a", "b", "c"]);

    // Non-object yields empty.
    let v = owned_json!(42);
    assert_eq!(v.entries().count(), 0);
}

#[test]
fn display_compact_matches_serde_json() {
    let v = owned_json!({
        "items": [1, 2, 3],
        "name": "x",
    });
    let ours = v.to_string();
    let serde: serde_json::Value = serde_json::from_str(&ours).unwrap();
    assert_eq!(ours, serde_json::to_string(&serde).unwrap());
}

#[test]
fn pretty_matches_serde_json_pretty() {
    // Use alphabetical key order so the test doesn't rely on `preserve_order`
    // in serde_json (which sorts keys by default).
    let v = owned_json!({
        "alpha": [1, 2, {"k": "v"}],
        "beta": "x",
        "empty_arr": [],
        "empty_obj": {},
    });
    let ours = v.pretty().to_string();
    let serde: serde_json::Value = serde_json::from_str(&v.to_string()).unwrap();
    assert_eq!(ours, serde_json::to_string_pretty(&serde).unwrap());
}

#[test]
fn arena_value_display_and_pretty() {
    use bumpalo::Bump;
    use datavalue_rs::DataValue;

    let arena = Bump::new();
    let v = DataValue::from_str(r#"{"a":[1,2]}"#, &arena).unwrap();
    assert_eq!(v.to_string(), r#"{"a":[1,2]}"#);
    assert_eq!(
        v.pretty().to_string(),
        "{\n  \"a\": [\n    1,\n    2\n  ]\n}"
    );

    // members / entries on the arena side
    let inner = &v["a"];
    let collected: Vec<i64> = inner.members().map(|m| m.as_i64().unwrap()).collect();
    assert_eq!(collected, vec![1, 2]);
    let keys: Vec<&str> = v.entries().map(|(k, _)| k).collect();
    assert_eq!(keys, vec!["a"]);
}
