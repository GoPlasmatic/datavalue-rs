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
mod value;

#[cfg(feature = "datetime")]
mod datetime;

#[cfg(feature = "serde")]
mod ser;

pub use number::NumberValue;
pub use owned::{OwnedDataValue, OwnedValueIndex};
pub use parser::{ParseError, ParseErrorKind};
pub use value::{DataValue, ValueIndex};

#[cfg(feature = "datetime")]
pub use datetime::{DataDateTime, DataDuration};

#[cfg(feature = "serde")]
pub use ser::DataValueSeed;
