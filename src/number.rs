//! `NumberValue` — numeric type for [`crate::ArenaValue`].
//!
//! Distinguishes `Integer(i64)` from `Float(f64)` natively (vs. the opaque
//! internal string of `serde_json::Number`) so integer arithmetic stays in
//! i64 with overflow checks instead of round-tripping through f64.

use std::cmp::Ordering;
use std::fmt;

/// Specialised numeric representation. Integers stay in i64 unless they
/// overflow during arithmetic, in which case the result falls back to f64.
#[derive(Debug, Clone, Copy)]
pub enum NumberValue {
    Integer(i64),
    Float(f64),
}

/// f64 -> i64 when the value is whole and exactly representable.
/// The upper bound is strict: `i64::MAX as f64` rounds up to 2^63, which
/// overflows i64 (a `<=` bound would admit 2^63 and saturate it to
/// `i64::MAX`, silently changing the value by 1). The largest admitted
/// value is 2^63 - 1024. `i64::MIN as f64` is exactly -2^63, so `>=` is
/// correct on the low side.
#[inline]
fn f64_as_i64_exact(value: f64) -> Option<i64> {
    if value.fract() == 0.0
        && !value.is_nan()
        && !value.is_infinite()
        && value >= i64::MIN as f64
        && value < i64::MAX as f64
    {
        Some(value as i64)
    } else {
        None
    }
}

/// Exclusive upper bound for a whole `f64` that fits in `u64`: exactly 2^64
/// (`u64::MAX as f64` rounds up to it, so `<` is the correct test).
const U64_LIMIT: f64 = 18_446_744_073_709_551_616.0;

impl NumberValue {
    #[inline]
    pub fn from_i64(value: i64) -> Self {
        NumberValue::Integer(value)
    }

    /// Construct from a `u64`. Values up to `i64::MAX` stay on the integer
    /// path; larger values fall back to `f64` — the same overflow rule the
    /// parser and the serde visitors apply. Bypasses `from_f64` so the
    /// fallback is not re-collapsed with saturation.
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        match i64::try_from(value) {
            Ok(i) => NumberValue::Integer(i),
            Err(_) => NumberValue::Float(value as f64),
        }
    }

    /// Construct from an f64. Whole-valued floats exactly representable in
    /// i64 collapse to `Integer` so subsequent arithmetic uses the integer
    /// fast path.
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        match f64_as_i64_exact(value) {
            Some(i) => NumberValue::Integer(i),
            None => NumberValue::Float(value),
        }
    }

    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(self, NumberValue::Integer(_))
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            NumberValue::Integer(i) => Some(i),
            NumberValue::Float(f) => f64_as_i64_exact(f),
        }
    }

    /// `u64` when the value is non-negative, whole, and below 2^64 — the
    /// unsigned twin of [`as_i64`](NumberValue::as_i64): `None` rather than
    /// an altered value when it does not fit.
    #[inline]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            NumberValue::Integer(i) => u64::try_from(i).ok(),
            NumberValue::Float(f) => {
                if f.fract() == 0.0 && (0.0..U64_LIMIT).contains(&f) {
                    Some(f as u64)
                } else {
                    None
                }
            }
        }
    }

    #[inline]
    pub fn as_f64(&self) -> f64 {
        match *self {
            NumberValue::Integer(i) => i as f64,
            NumberValue::Float(f) => f,
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        match *self {
            NumberValue::Integer(i) => i == 0,
            NumberValue::Float(f) => f == 0.0,
        }
    }

    #[inline]
    pub fn is_nan(&self) -> bool {
        matches!(*self, NumberValue::Float(f) if f.is_nan())
    }

    /// Add. Integer-integer uses checked_add; on overflow falls back to f64.
    pub fn add(&self, other: &NumberValue) -> NumberValue {
        match (*self, *other) {
            (NumberValue::Integer(a), NumberValue::Integer(b)) => match a.checked_add(b) {
                Some(r) => NumberValue::Integer(r),
                None => NumberValue::Float(a as f64 + b as f64),
            },
            _ => NumberValue::from_f64(self.as_f64() + other.as_f64()),
        }
    }

    pub fn sub(&self, other: &NumberValue) -> NumberValue {
        match (*self, *other) {
            (NumberValue::Integer(a), NumberValue::Integer(b)) => match a.checked_sub(b) {
                Some(r) => NumberValue::Integer(r),
                None => NumberValue::Float(a as f64 - b as f64),
            },
            _ => NumberValue::from_f64(self.as_f64() - other.as_f64()),
        }
    }

    pub fn mul(&self, other: &NumberValue) -> NumberValue {
        match (*self, *other) {
            (NumberValue::Integer(a), NumberValue::Integer(b)) => match a.checked_mul(b) {
                Some(r) => NumberValue::Integer(r),
                None => NumberValue::Float(a as f64 * b as f64),
            },
            _ => NumberValue::from_f64(self.as_f64() * other.as_f64()),
        }
    }

    /// Divide. Returns `None` for division by zero — callers handle.
    pub fn div(&self, other: &NumberValue) -> Option<NumberValue> {
        if other.is_zero() {
            return None;
        }
        match (*self, *other) {
            (NumberValue::Integer(a), NumberValue::Integer(b)) => {
                // i64::MIN / -1 overflows; fall through to float.
                if a == i64::MIN && b == -1 {
                    return Some(NumberValue::Float(-(i64::MIN as f64)));
                }
                if a % b == 0 {
                    Some(NumberValue::Integer(a / b))
                } else {
                    Some(NumberValue::Float(a as f64 / b as f64))
                }
            }
            _ => Some(NumberValue::from_f64(self.as_f64() / other.as_f64())),
        }
    }

    /// Modulo. Returns `None` for division by zero — caller handles.
    pub fn rem(&self, other: &NumberValue) -> Option<NumberValue> {
        if other.is_zero() {
            return None;
        }
        match (*self, *other) {
            (NumberValue::Integer(a), NumberValue::Integer(b)) => {
                // i64::MIN % -1 overflows; the mathematical result is 0.
                if a == i64::MIN && b == -1 {
                    return Some(NumberValue::Integer(0));
                }
                Some(NumberValue::Integer(a % b))
            }
            _ => Some(NumberValue::from_f64(self.as_f64() % other.as_f64())),
        }
    }

    pub fn neg(&self) -> NumberValue {
        match *self {
            NumberValue::Integer(i) => match i.checked_neg() {
                Some(r) => NumberValue::Integer(r),
                None => NumberValue::Float(-(i as f64)),
            },
            NumberValue::Float(f) => NumberValue::Float(-f),
        }
    }

    pub fn abs(&self) -> NumberValue {
        match *self {
            NumberValue::Integer(i) => match i.checked_abs() {
                Some(r) => NumberValue::Integer(r),
                None => NumberValue::Float((i as f64).abs()),
            },
            NumberValue::Float(f) => NumberValue::Float(f.abs()),
        }
    }
}

impl PartialEq for NumberValue {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (NumberValue::Integer(a), NumberValue::Integer(b)) => a == b,
            (NumberValue::Float(a), NumberValue::Float(b)) => a == b,
            (NumberValue::Integer(a), NumberValue::Float(b)) => (a as f64) == b,
            (NumberValue::Float(a), NumberValue::Integer(b)) => a == (b as f64),
        }
    }
}

impl PartialOrd for NumberValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (*self, *other) {
            (NumberValue::Integer(a), NumberValue::Integer(b)) => Some(a.cmp(&b)),
            (NumberValue::Float(a), NumberValue::Float(b)) => a.partial_cmp(&b),
            (NumberValue::Integer(a), NumberValue::Float(b)) => (a as f64).partial_cmp(&b),
            (NumberValue::Float(a), NumberValue::Integer(b)) => a.partial_cmp(&(b as f64)),
        }
    }
}

impl fmt::Display for NumberValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            NumberValue::Integer(i) => write!(f, "{}", i),
            NumberValue::Float(fl) => {
                // Match serde_json::Number's f64 formatting: "1.5" not "1.5e0".
                if fl.is_nan() || fl.is_infinite() {
                    write!(f, "null")
                } else if let Some(i) = f64_as_i64_exact(fl) {
                    write!(f, "{}.0", i)
                } else if fl.fract() == 0.0 {
                    // Whole float outside i64's exact range: {:?} keeps the
                    // float shape ("9.223372036854776e18"), matching
                    // ryu/serde_json for these magnitudes.
                    write!(f, "{:?}", fl)
                } else {
                    write!(f, "{}", fl)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_collapses_whole() {
        assert!(matches!(
            NumberValue::from_f64(42.0),
            NumberValue::Integer(42)
        ));
        assert!(matches!(
            NumberValue::from_f64(-3.0),
            NumberValue::Integer(-3)
        ));
        assert!(matches!(NumberValue::from_f64(1.5), NumberValue::Float(_)));
    }

    #[test]
    fn from_f64_rejects_nan_inf_for_int_path() {
        assert!(matches!(
            NumberValue::from_f64(f64::NAN),
            NumberValue::Float(_)
        ));
        assert!(matches!(
            NumberValue::from_f64(f64::INFINITY),
            NumberValue::Float(_)
        ));
    }

    // 2^63: what `i64::MAX as f64` actually rounds up to. Not representable
    // as i64, so it must stay Float.
    const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;
    // Largest f64 below 2^63 (= 2^63 - 1024): the biggest float that
    // converts to i64 exactly.
    const BELOW_TWO_POW_63: f64 = 9_223_372_036_854_774_784.0;

    #[test]
    fn from_f64_boundary_at_two_pow_63() {
        assert!(matches!(
            NumberValue::from_f64(TWO_POW_63),
            NumberValue::Float(f) if f == TWO_POW_63
        ));
        assert!(matches!(
            NumberValue::from_f64(BELOW_TWO_POW_63),
            NumberValue::Integer(9_223_372_036_854_774_784)
        ));
        // i64::MIN is exactly -2^63, representable, so it collapses.
        assert!(matches!(
            NumberValue::from_f64(i64::MIN as f64),
            NumberValue::Integer(i64::MIN)
        ));
    }

    #[test]
    fn as_i64_float_boundary_at_two_pow_63() {
        assert_eq!(NumberValue::Float(TWO_POW_63).as_i64(), None);
        assert_eq!(
            NumberValue::Float(BELOW_TWO_POW_63).as_i64(),
            Some(9_223_372_036_854_774_784)
        );
        assert_eq!(NumberValue::Float(i64::MIN as f64).as_i64(), Some(i64::MIN));
    }

    #[test]
    fn display_whole_float_beyond_i64_range() {
        // Previously the unguarded `as i64` cast saturated, printing the
        // off-by-one "9223372036854775807.0".
        assert_eq!(
            NumberValue::Float(TWO_POW_63).to_string(),
            "9.223372036854776e18"
        );
        assert_eq!(
            NumberValue::Float(BELOW_TWO_POW_63).to_string(),
            "9223372036854774784.0"
        );
    }

    #[test]
    fn add_overflow_falls_to_float() {
        let a = NumberValue::Integer(i64::MAX);
        let b = NumberValue::Integer(1);
        assert!(matches!(a.add(&b), NumberValue::Float(_)));
    }

    #[test]
    fn add_no_overflow_stays_int() {
        let a = NumberValue::Integer(2);
        let b = NumberValue::Integer(3);
        assert!(matches!(a.add(&b), NumberValue::Integer(5)));
    }

    #[test]
    fn div_zero_returns_none() {
        let a = NumberValue::Integer(1);
        let z = NumberValue::Integer(0);
        assert!(a.div(&z).is_none());
        let zf = NumberValue::Float(0.0);
        assert!(a.div(&zf).is_none());
    }

    #[test]
    fn div_int_int_exact_stays_int() {
        let a = NumberValue::Integer(10);
        let b = NumberValue::Integer(2);
        assert!(matches!(a.div(&b).unwrap(), NumberValue::Integer(5)));
    }

    #[test]
    fn div_int_int_inexact_promotes_float() {
        let a = NumberValue::Integer(7);
        let b = NumberValue::Integer(2);
        assert!(matches!(a.div(&b).unwrap(), NumberValue::Float(_)));
    }

    #[test]
    fn cross_type_eq_and_ord() {
        let i = NumberValue::Integer(5);
        let f = NumberValue::Float(5.0);
        assert_eq!(i, f);
        assert_eq!(i.partial_cmp(&f), Some(Ordering::Equal));

        let f2 = NumberValue::Float(5.5);
        assert_eq!(i.partial_cmp(&f2), Some(Ordering::Less));
    }

    #[test]
    fn neg_overflow_falls_to_float() {
        let a = NumberValue::Integer(i64::MIN);
        assert!(matches!(a.neg(), NumberValue::Float(_)));
    }

    #[test]
    fn u64_round_trips_through_from_u64_and_as_u64() {
        assert_eq!(NumberValue::from_u64(7), NumberValue::Integer(7));
        assert_eq!(NumberValue::from_u64(7).as_u64(), Some(7));
        let big = NumberValue::from_u64(1u64 << 63);
        assert!(matches!(big, NumberValue::Float(_)));
        assert_eq!(big.as_u64(), Some(1u64 << 63));
        assert_eq!(NumberValue::Integer(-1).as_u64(), None);
        assert_eq!(NumberValue::Float(1.5).as_u64(), None);
        assert_eq!(NumberValue::Float(-0.0).as_u64(), Some(0));
        assert_eq!(
            NumberValue::Float(18_446_744_073_709_551_616.0).as_u64(),
            None
        );
        assert_eq!(NumberValue::Float(f64::NAN).as_u64(), None);
    }
}
