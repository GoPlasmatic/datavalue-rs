//! Public-API integration tests for the `tensor` feature: the four data
//! paths from the proposal (nested JSON arrays, the tagged base64 form,
//! binary buffers, element-wise construction) and the ONNX-style typed
//! read-out, exercised only through the crate's exported surface.

#![cfg(feature = "tensor")]

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use datavalue_rs::{
    DType, DataTensor, DataValue, MAX_RANK, OwnedDataTensor, OwnedDataValue, TensorError,
};

#[test]
fn path_nested_json_arrays_to_typed_input() {
    let arena = Bump::new();
    let request = DataValue::from_str(r#"{"pixels":[[0,128,255],[1,2,3]]}"#, &arena).unwrap();
    let t = DataTensor::from_nested_in(&request["pixels"], DType::F32, &arena).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    let input: &[f32] = t.as_slice().unwrap();
    assert_eq!(input, &[0.0, 128.0, 255.0, 1.0, 2.0, 3.0]);
    assert_eq!(input.as_ptr() as usize % 4, 0);
}

#[test]
fn path_tagged_base64_request_and_response() {
    let arena = Bump::new();
    let wire = r#"{"input":{"tensor":{"dtype":"i64","shape":[3],"data":"AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA"}}}"#;
    let doc = DataValue::from_str(wire, &arena).unwrap();
    assert!(doc["input"].is_object(), "the parser never upgrades");
    let t = DataTensor::from_json_value_in(&doc["input"], &arena).unwrap();
    assert_eq!(t.as_slice::<i64>().unwrap(), &[1, 2, 3]);

    // A response carries a tensor straight back out in the same form.
    let out = arena.alloc_slice_copy(&[("output", DataValue::tensor_in(t, &arena))]);
    let rendered = DataValue::Object(out).to_string();
    assert_eq!(
        rendered,
        r#"{"output":{"tensor":{"dtype":"i64","shape":[3],"data":"AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA"}}}"#
    );
    let reparsed = DataValue::from_str(&rendered, &arena).unwrap();
    assert_eq!(
        DataTensor::from_json_value_in(&reparsed["output"], &arena).unwrap(),
        t
    );
}

#[test]
fn path_binary_buffer_wrapped_without_copy() {
    // A buffer that already exists (an mmap, a runtime output, an Arc'd
    // cache) is wrapped in place when it is aligned, and rejected when not.
    let arena = Bump::new();
    let backing: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5];
    let shape = arena.alloc_slice_copy(&[2, 2]);
    let t = DataTensor::from_slice(shape, &backing).unwrap();
    assert_eq!(t.data().as_ptr(), backing.as_ptr().cast::<u8>());
    assert_eq!(t.as_slice::<f64>().unwrap().as_ptr(), backing.as_ptr());

    #[repr(align(8))]
    struct Aligned([u8; 40]);
    let raw = Aligned([0; 40]);
    assert_eq!(
        DataTensor::from_bytes(DType::F64, shape, &raw.0[1..33]),
        Err(TensorError::Misaligned { required: 8 })
    );
    // The copying constructor fixes it up.
    let fixed = DataTensor::from_bytes_in(DType::F64, shape, &raw.0[1..33], &arena).unwrap();
    assert_eq!(fixed.as_slice::<f64>().unwrap(), &[0.0; 4]);
}

#[test]
fn path_element_wise_construction_via_bump_vec() {
    let arena = Bump::new();
    let mut acc: BumpVec<'_, u16> = BumpVec::with_capacity_in(6, &arena);
    for i in 0..6u16 {
        acc.push(i * 100);
    }
    let elems = acc.into_bump_slice();
    let shape = arena.alloc_slice_copy(&[3, 2]);
    let t = DataTensor::from_slice(shape, elems).unwrap();
    assert_eq!(t.data().as_ptr(), elems.as_ptr().cast::<u8>(), "zero-copy");
    let v = DataValue::tensor_in(t, &arena);
    assert_eq!(
        v.to_string(),
        r#"{"tensor":{"dtype":"u16","shape":[3,2],"data":"AABkAMgALAGQAfQB"}}"#
    );
}

#[test]
fn read_back_expands_to_plain_arrays_on_request() {
    let arena = Bump::new();
    let t = DataTensor::from_slice_in(&[2, 2], &[true, false, false, true], &arena).unwrap();
    let nested = t.to_nested_in(&arena).unwrap();
    assert_eq!(nested.to_string(), "[[true,false],[false,true]]");
    let owned = OwnedDataTensor::from_nested(&nested.to_owned(), DType::Bool).unwrap();
    assert_eq!(
        owned.to_nested().unwrap().to_string(),
        "[[true,false],[false,true]]"
    );
}

#[test]
fn owned_values_share_tensors_and_escape_the_arena() {
    let owned = {
        let arena = Bump::new();
        let t = DataTensor::from_slice_in(&[2], &[1i32, -1], &arena).unwrap();
        DataValue::tensor_in(t, &arena).to_owned()
    };
    let json: OwnedDataValue = owned.to_string().parse().unwrap();
    assert!(json.is_object());
    assert_eq!(
        OwnedDataTensor::try_from(&json).unwrap().as_slice::<i32>(),
        Some(&[1i32, -1][..])
    );
    let shared = std::sync::Arc::new(OwnedDataTensor::from_slice([2], &[1i32, -1]).unwrap());
    let a = OwnedDataValue::from(shared.clone());
    let b = OwnedDataValue::from(shared.clone());
    assert_eq!(a, b);
    assert_eq!(a, owned);
    assert_eq!(std::sync::Arc::strong_count(&shared), 3);
}

#[test]
fn dtype_is_a_first_class_public_type() {
    assert_eq!("BF16".parse::<DType>().unwrap(), DType::BF16);
    assert_eq!(DType::BF16.size_of(), 2);
    assert_eq!(DType::I64.byte_len(4), Some(32));
    assert_eq!(format!("{}", DType::F64), "f64");
    // F16 / BF16 carry bytes but have no typed view without `tensor-half`.
    let t = OwnedDataTensor::from_bytes(DType::BF16, [2], &[0x80, 0x3F, 0x00, 0x40]).unwrap();
    assert_eq!(t.data(), &[0x80, 0x3F, 0x00, 0x40]);
    assert!(t.as_slice::<u16>().is_none(), "no cross-dtype views");
}

#[test]
fn rank_is_capped_on_every_construction_path() {
    let arena = Bump::new();

    // A shape of 1s costs one element and one byte, so nothing but the rank
    // cap stands between a wire payload and a recursion as deep as its shape.
    let dims = vec!["1"; MAX_RANK + 1].join(",");
    let wire = format!(r#"{{"tensor":{{"dtype":"u8","shape":[{dims}],"data":"AA=="}}}}"#);
    let doc = DataValue::from_str(&wire, &arena).unwrap();
    assert!(matches!(
        DataTensor::from_json_value_in(&doc, &arena),
        Err(TensorError::RankTooHigh { max, actual }) if max == MAX_RANK && actual == MAX_RANK + 1
    ));

    // Wrapping caller-owned bytes is checked the same way...
    assert!(matches!(
        OwnedDataTensor::from_bytes(DType::U8, vec![1usize; MAX_RANK + 1], &[0]),
        Err(TensorError::RankTooHigh { .. })
    ));

    // ...as is a nested tree deeper than the parser would ever hand over.
    let mut nested = DataValue::from_str("0", &arena).unwrap();
    for _ in 0..MAX_RANK + 1 {
        nested = DataValue::Array(arena.alloc_slice_copy(&[nested]));
    }
    assert!(matches!(
        DataTensor::from_nested_in(&nested, DType::U8, &arena),
        Err(TensorError::RankTooHigh { .. })
    ));

    // At the cap everything still works, expansion included.
    let at_cap = OwnedDataTensor::from_bytes(DType::U8, vec![1usize; MAX_RANK], &[7]).unwrap();
    assert_eq!(at_cap.shape().len(), MAX_RANK);
    let expanded = at_cap.to_nested().unwrap();
    assert_eq!(
        expanded.to_string(),
        format!("{}7{}", "[".repeat(MAX_RANK), "]".repeat(MAX_RANK))
    );
}

#[test]
fn tagged_decoder_rejects_duplicate_inner_keys() {
    let arena = Bump::new();
    for (dup, wire) in [
        (
            "dtype",
            r#"{"tensor":{"dtype":"u8","dtype":"i8","shape":[1],"data":"AA=="}}"#,
        ),
        (
            "shape",
            r#"{"tensor":{"dtype":"u8","shape":[1],"shape":[1],"data":"AA=="}}"#,
        ),
        (
            "data",
            r#"{"tensor":{"dtype":"u8","shape":[1],"data":"AA==","data":"AQ=="}}"#,
        ),
    ] {
        let doc = DataValue::from_str(wire, &arena).unwrap();
        assert!(
            matches!(
                DataTensor::from_json_value_in(&doc, &arena),
                Err(TensorError::UnexpectedField(ref k)) if k == dup
            ),
            "duplicate {dup:?} must not last-win"
        );
    }
}

#[test]
fn expanded_arrays_outlive_the_tensor_they_came_from() {
    let arena = Bump::new();
    let expanded = {
        let owned = OwnedDataTensor::from_slice([2, 2], &[1i32, 2, 3, 4]).unwrap();
        owned.view().to_nested_in(&arena).unwrap()
    }; // `owned` is gone; every element was copied into the arena.
    assert_eq!(expanded.to_string(), "[[1,2],[3,4]]");
}
