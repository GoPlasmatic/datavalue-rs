//! Native JSON emitter for [`DataValue`] and [`OwnedDataValue`].
//!
//! Bypasses the `serde_json::to_string` path. The serde route pays trait
//! dispatch per node and a per-byte string-escape loop; emitting directly
//! into a `Vec<u8>` buffer with `ryu` / `itoa` and a SWAR-driven escape
//! scan lands closer to the bespoke emitters in `json-rust` / `simd_json`.
//!
//! The `Serialize` impl in [`crate::ser`] is still the right entry point
//! when feeding non-JSON serde sinks (msgpack, flexbuffers, etc.).

use crate::number::NumberValue;
use crate::owned::OwnedDataValue;
use crate::value::DataValue;

const SWAR_ONES: u64 = 0x0101_0101_0101_0101;
const SWAR_HIGHS: u64 = 0x8080_8080_8080_8080;

/// SWAR scan for the next byte that needs escaping inside a JSON string:
/// `"`, `\\`, or any control byte (< 0x20). Mirrors the parser's scan.
#[inline(always)]
fn escape_mask(w: u64) -> u64 {
    let q = w ^ (b'"' as u64 * SWAR_ONES);
    let bs = w ^ (b'\\' as u64 * SWAR_ONES);
    let lo = w & 0xE0E0_E0E0_E0E0_E0E0;
    let m_q = q.wrapping_sub(SWAR_ONES) & !q;
    let m_bs = bs.wrapping_sub(SWAR_ONES) & !bs;
    let m_lo = lo.wrapping_sub(SWAR_ONES) & !lo;
    (m_q | m_bs | m_lo) & SWAR_HIGHS
}

#[inline]
fn write_escaped_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut run_start = 0;

    while i + 8 <= bytes.len() {
        let w = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let mask = escape_mask(w);
        if mask == 0 {
            i += 8;
            continue;
        }
        let off = (mask.trailing_zeros() / 8) as usize;
        let hit = i + off;
        if hit > run_start {
            out.extend_from_slice(&bytes[run_start..hit]);
        }
        write_escape_byte(out, bytes[hit]);
        i = hit + 1;
        run_start = i;
    }
    // Tail: per-byte for the final < 8 bytes.
    while i < bytes.len() {
        let b = bytes[i];
        if matches!(b, b'"' | b'\\') || b < 0x20 {
            if i > run_start {
                out.extend_from_slice(&bytes[run_start..i]);
            }
            write_escape_byte(out, b);
            run_start = i + 1;
        }
        i += 1;
    }
    if run_start < bytes.len() {
        out.extend_from_slice(&bytes[run_start..]);
    }
    out.push(b'"');
}

#[inline]
fn write_escape_byte(out: &mut Vec<u8>, b: u8) {
    match b {
        b'"' => out.extend_from_slice(b"\\\""),
        b'\\' => out.extend_from_slice(b"\\\\"),
        b'\n' => out.extend_from_slice(b"\\n"),
        b'\r' => out.extend_from_slice(b"\\r"),
        b'\t' => out.extend_from_slice(b"\\t"),
        0x08 => out.extend_from_slice(b"\\b"),
        0x0C => out.extend_from_slice(b"\\f"),
        c => {
            // Other control bytes (< 0x20 not named above) use \u00XX. The
            // high byte is always 0 here.
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let prefix = [b'\\', b'u', b'0', b'0'];
            out.extend_from_slice(&prefix);
            out.push(HEX[((c >> 4) & 0x0F) as usize]);
            out.push(HEX[(c & 0x0F) as usize]);
        }
    }
}

#[inline]
fn write_number(out: &mut Vec<u8>, n: NumberValue) {
    match n {
        NumberValue::Integer(i) => {
            let mut buf = itoa::Buffer::new();
            out.extend_from_slice(buf.format(i).as_bytes());
        }
        NumberValue::Float(f) => {
            if !f.is_finite() {
                // serde_json emits non-finite floats as `null` to keep
                // output valid JSON. Match that.
                out.extend_from_slice(b"null");
                return;
            }
            let mut buf = ryu::Buffer::new();
            out.extend_from_slice(buf.format_finite(f).as_bytes());
        }
    }
}

fn write_data_value(out: &mut Vec<u8>, v: &DataValue<'_>) {
    match *v {
        DataValue::Null => out.extend_from_slice(b"null"),
        DataValue::Bool(true) => out.extend_from_slice(b"true"),
        DataValue::Bool(false) => out.extend_from_slice(b"false"),
        DataValue::Number(n) => write_number(out, n),
        DataValue::String(s) => write_escaped_str(out, s),
        DataValue::Array(items) => {
            out.push(b'[');
            let mut first = true;
            for item in items {
                if !first {
                    out.push(b',');
                }
                first = false;
                write_data_value(out, item);
            }
            out.push(b']');
        }
        DataValue::Object(pairs) => {
            out.push(b'{');
            let mut first = true;
            for (k, v) in pairs {
                if !first {
                    out.push(b',');
                }
                first = false;
                write_escaped_str(out, k);
                out.push(b':');
                write_data_value(out, v);
            }
            out.push(b'}');
        }
        #[cfg(feature = "datetime")]
        DataValue::DateTime(d) => write_escaped_str(out, &d.to_iso_string()),
        #[cfg(feature = "datetime")]
        DataValue::Duration(d) => write_escaped_str(out, &d.to_string()),
    }
}

fn write_owned_value(out: &mut Vec<u8>, v: &OwnedDataValue) {
    match v {
        OwnedDataValue::Null => out.extend_from_slice(b"null"),
        OwnedDataValue::Bool(true) => out.extend_from_slice(b"true"),
        OwnedDataValue::Bool(false) => out.extend_from_slice(b"false"),
        OwnedDataValue::Number(n) => write_number(out, *n),
        OwnedDataValue::String(s) => write_escaped_str(out, s),
        OwnedDataValue::Array(items) => {
            out.push(b'[');
            let mut first = true;
            for item in items {
                if !first {
                    out.push(b',');
                }
                first = false;
                write_owned_value(out, item);
            }
            out.push(b']');
        }
        OwnedDataValue::Object(pairs) => {
            out.push(b'{');
            let mut first = true;
            for (k, v) in pairs {
                if !first {
                    out.push(b',');
                }
                first = false;
                write_escaped_str(out, k);
                out.push(b':');
                write_owned_value(out, v);
            }
            out.push(b'}');
        }
        #[cfg(feature = "datetime")]
        OwnedDataValue::DateTime(d) => write_escaped_str(out, &d.to_iso_string()),
        #[cfg(feature = "datetime")]
        OwnedDataValue::Duration(d) => write_escaped_str(out, &d.to_string()),
    }
}

impl DataValue<'_> {
    /// Serialize this value to a compact JSON string. Faster than going
    /// through `serde_json::to_string` because it skips serde dispatch
    /// and uses `ryu` / `itoa` directly.
    ///
    /// ```
    /// use bumpalo::Bump;
    /// use datavalue_rs::DataValue;
    ///
    /// let arena = Bump::new();
    /// let v = DataValue::from_str(r#"{"a":[1,2.5,"hi"]}"#, &arena).unwrap();
    /// assert_eq!(v.to_json_string(), r#"{"a":[1,2.5,"hi"]}"#);
    /// ```
    pub fn to_json_string(&self) -> String {
        let mut out = Vec::with_capacity(64);
        write_data_value(&mut out, self);
        // SAFETY: every byte we wrote came from a valid &str (passed through
        // unmodified) or is ASCII (numbers, escapes, structural punctuation).
        unsafe { String::from_utf8_unchecked(out) }
    }

    /// Append the JSON encoding of this value to `out`.
    pub fn write_json_into(&self, out: &mut Vec<u8>) {
        write_data_value(out, self);
    }
}

impl OwnedDataValue {
    /// Serialize this value to a compact JSON string. See
    /// [`DataValue::to_json_string`] for the rationale; this is the
    /// owned-side mirror.
    ///
    /// ```
    /// use datavalue_rs::OwnedDataValue;
    ///
    /// let v: OwnedDataValue = r#"{"a":[1,2.5,"hi"]}"#.parse().unwrap();
    /// assert_eq!(v.to_json_string(), r#"{"a":[1,2.5,"hi"]}"#);
    /// ```
    pub fn to_json_string(&self) -> String {
        let mut out = Vec::with_capacity(64);
        write_owned_value(&mut out, self);
        unsafe { String::from_utf8_unchecked(out) }
    }

    /// Append the JSON encoding of this value to `out`.
    pub fn write_json_into(&self, out: &mut Vec<u8>) {
        write_owned_value(out, self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    fn round_trip(s: &str) -> String {
        let arena = Bump::new();
        let v = DataValue::from_str(s, &arena).unwrap();
        v.to_json_string()
    }

    #[test]
    fn primitives() {
        assert_eq!(round_trip("null"), "null");
        assert_eq!(round_trip("true"), "true");
        assert_eq!(round_trip("false"), "false");
        assert_eq!(round_trip("42"), "42");
        assert_eq!(round_trip("-7"), "-7");
        assert_eq!(round_trip("3.5"), "3.5");
    }

    #[test]
    fn strings_with_escapes() {
        assert_eq!(round_trip(r#""hello""#), r#""hello""#);
        assert_eq!(round_trip(r#""a\nb""#), r#""a\nb""#);
        assert_eq!(round_trip(r#""a\\b""#), r#""a\\b""#);
        assert_eq!(round_trip(r#""a\"b""#), r#""a\"b""#);
        // Unicode passes through verbatim (we don't re-escape non-ASCII).
        assert_eq!(round_trip(r#""café""#), r#""café""#);
    }

    #[test]
    fn control_bytes_render_as_unicode_escapes() {
        let arena = Bump::new();
        let v = DataValue::from_str("\"\\u0001\"", &arena).unwrap();
        assert_eq!(v.to_json_string(), "\"\\u0001\"");
    }

    #[test]
    fn nested_round_trip_matches_serde_json() {
        let input =
            r#"{"a":[1,2,{"b":"hi\n","c":null,"d":true}],"e":-3.5,"f":[],"g":{}}"#;
        let arena = Bump::new();
        let v = DataValue::from_str(input, &arena).unwrap();
        let ours = v.to_json_string();
        let serde: serde_json::Value = serde_json::from_str(input).unwrap();
        let theirs = serde_json::to_string(&serde).unwrap();
        assert_eq!(ours, theirs);
    }

    #[test]
    fn long_string_swar_path() {
        let arena = Bump::new();
        let s = format!("\"{}\"", "x".repeat(200));
        let v = DataValue::from_str(&s, &arena).unwrap();
        assert_eq!(v.to_json_string(), s);
    }

    #[test]
    fn non_finite_floats_render_as_null() {
        let v = DataValue::from_f64(f64::NAN);
        assert_eq!(v.to_json_string(), "null");
        let v = DataValue::from_f64(f64::INFINITY);
        assert_eq!(v.to_json_string(), "null");
    }

    #[test]
    fn owned_round_trip() {
        let v: OwnedDataValue = r#"{"name":"alice","age":30}"#.parse().unwrap();
        let serde: serde_json::Value = serde_json::from_str(&v.to_json_string()).unwrap();
        assert_eq!(serde["name"], "alice");
        assert_eq!(serde["age"], 30);
    }
}
