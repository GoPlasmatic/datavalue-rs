//! datavalue-rs — bump-allocated JSON value type.
//!
//! [`DataValue<'a>`] mirrors the shape of `serde_json::Value`, but every
//! composite payload (string bytes, array elements, object pairs) lives in
//! a [`bumpalo::Bump`] arena. Designed for hot paths where per-value heap
//! allocation is the bottleneck and the same arena handles many values
//! before being reset.
//!
//! ## Quickstart
//!
//! ```
//! use bumpalo::Bump;
//! use datavalue_rs::DataValue;
//!
//! let arena = Bump::new();
//! let v = DataValue::from_str(r#"{"name":"alice","ages":[30,31]}"#, &arena).unwrap();
//! assert_eq!(v["name"].as_str(), Some("alice"));
//! assert_eq!(v["ages"][1].as_i64(), Some(31));
//! ```

mod emit;
mod number;
mod owned;
mod parser;
pub mod simd;
mod value;

#[cfg(feature = "datetime")]
mod datetime;

#[cfg(feature = "serde")]
mod ser;

#[cfg(feature = "serde_json")]
mod serde_json_bridge;

pub use number::NumberValue;
pub use owned::{OwnedDataValue, OwnedValueIndex};
pub use parser::{ParseError, ParseErrorKind};
pub use value::{DataValue, ValueIndex};

#[cfg(feature = "datetime")]
pub use datetime::{DataDateTime, DataDuration};

#[cfg(feature = "serde")]
pub use ser::DataValueSeed;

pub use emit::Pretty;

/// Construct an [`OwnedDataValue`] from a JSON-shaped literal.
///
/// Modeled on `serde_json::json!`. Each `null`, `true`, `false`, array,
/// or object literal maps to the corresponding variant; any other token
/// is forwarded through `OwnedDataValue::from(...)` so the [`From`] impls
/// on this crate (`i32`, `String`, `Vec<T>`, `Option<T>`, `HashMap`, etc.)
/// determine the variant.
///
/// Each array element and each object value must be a single token tree.
/// Non-trivial expressions need to be parenthesised:
/// `owned_json!([(1 + 2), (compute())])`.
///
/// ```
/// use datavalue_rs::owned_json;
///
/// let v = owned_json!({
///     "name": "alice",
///     "ages": [30, 31],
///     "active": true,
///     "tags": null,
/// });
/// assert_eq!(v["name"].as_str(), Some("alice"));
/// assert_eq!(v["ages"][1].as_i64(), Some(31));
/// ```
#[macro_export]
macro_rules! owned_json {
    (null) => { $crate::OwnedDataValue::Null };
    (true) => { $crate::OwnedDataValue::Bool(true) };
    (false) => { $crate::OwnedDataValue::Bool(false) };

    ([]) => { $crate::OwnedDataValue::Array(::std::vec::Vec::new()) };
    ([ $( $elem:tt ),+ $(,)? ]) => {
        $crate::OwnedDataValue::Array(::std::vec![
            $( $crate::owned_json!($elem) ),+
        ])
    };

    ({}) => { $crate::OwnedDataValue::Object(::std::vec::Vec::new()) };
    ({ $( $key:tt : $val:tt ),+ $(,)? }) => {
        $crate::OwnedDataValue::Object(::std::vec![
            $( (
                ::std::string::ToString::to_string(&$key),
                $crate::owned_json!($val),
            ) ),+
        ])
    };

    ($other:expr) => { $crate::OwnedDataValue::from($other) };
}
