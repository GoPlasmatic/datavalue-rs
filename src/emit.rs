//! Native JSON emitter for [`DataValue`] and [`OwnedDataValue`].
//!
//! Bypasses the `serde_json::to_string` path. The serde route pays trait
//! dispatch per node and a per-byte string-escape loop; emitting directly
//! into a buffer with `ryu` / `itoa` and a SWAR-driven escape scan lands
//! closer to the bespoke emitters in `json-rust` / `simd_json`.
//!
//! The same writers feed three sinks — `Vec<u8>` for [`DataValue::write_json_into`],
//! `fmt::Formatter` for the [`fmt::Display`] impls, and an indenting wrapper
//! for [`DataValue::pretty`] — through the [`JsonSink`] trait below.
//!
//! The `Serialize` impl in [`crate::ser`] is still the right entry point
//! when feeding non-JSON serde sinks (msgpack, flexbuffers, etc.).

use core::fmt;

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

/// Sink abstraction over `Vec<u8>` and `fmt::Formatter`. Bytes pushed are
/// always valid UTF-8 (numbers are ASCII; strings are passed through from
/// `&str` sources; escapes are ASCII), so the str adapter is sound.
pub(crate) trait JsonSink {
    type Error;
    fn write_bytes(&mut self, b: &[u8]) -> Result<(), Self::Error>;
    fn write_byte(&mut self, b: u8) -> Result<(), Self::Error>;
}

impl JsonSink for Vec<u8> {
    type Error = core::convert::Infallible;
    #[inline]
    fn write_bytes(&mut self, b: &[u8]) -> Result<(), Self::Error> {
        self.extend_from_slice(b);
        Ok(())
    }
    #[inline]
    fn write_byte(&mut self, b: u8) -> Result<(), Self::Error> {
        self.push(b);
        Ok(())
    }
}

struct FormatterSink<'a, 'b>(&'a mut fmt::Formatter<'b>);

impl<'a, 'b> JsonSink for FormatterSink<'a, 'b> {
    type Error = fmt::Error;
    #[inline]
    fn write_bytes(&mut self, b: &[u8]) -> Result<(), Self::Error> {
        // SAFETY: every caller writes either ASCII bytes (escapes, numbers,
        // structural punctuation) or pre-validated `&str` payloads.
        let s = unsafe { core::str::from_utf8_unchecked(b) };
        self.0.write_str(s)
    }
    #[inline]
    fn write_byte(&mut self, b: u8) -> Result<(), Self::Error> {
        debug_assert!(b.is_ascii());
        let buf = [b];
        let s = unsafe { core::str::from_utf8_unchecked(&buf) };
        self.0.write_str(s)
    }
}

#[inline]
fn write_escaped_str<S: JsonSink>(out: &mut S, s: &str) -> Result<(), S::Error> {
    out.write_byte(b'"')?;
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
            out.write_bytes(&bytes[run_start..hit])?;
        }
        write_escape_byte(out, bytes[hit])?;
        i = hit + 1;
        run_start = i;
    }
    // Tail: per-byte for the final < 8 bytes.
    while i < bytes.len() {
        let b = bytes[i];
        if matches!(b, b'"' | b'\\') || b < 0x20 {
            if i > run_start {
                out.write_bytes(&bytes[run_start..i])?;
            }
            write_escape_byte(out, b)?;
            run_start = i + 1;
        }
        i += 1;
    }
    if run_start < bytes.len() {
        out.write_bytes(&bytes[run_start..])?;
    }
    out.write_byte(b'"')
}

#[inline]
fn write_escape_byte<S: JsonSink>(out: &mut S, b: u8) -> Result<(), S::Error> {
    match b {
        b'"' => out.write_bytes(b"\\\""),
        b'\\' => out.write_bytes(b"\\\\"),
        b'\n' => out.write_bytes(b"\\n"),
        b'\r' => out.write_bytes(b"\\r"),
        b'\t' => out.write_bytes(b"\\t"),
        0x08 => out.write_bytes(b"\\b"),
        0x0C => out.write_bytes(b"\\f"),
        c => {
            // Other control bytes (< 0x20 not named above) use \u00XX. The
            // high byte is always 0 here.
            const HEX: &[u8; 16] = b"0123456789abcdef";
            out.write_bytes(b"\\u00")?;
            out.write_byte(HEX[((c >> 4) & 0x0F) as usize])?;
            out.write_byte(HEX[(c & 0x0F) as usize])
        }
    }
}

#[inline]
fn write_number<S: JsonSink>(out: &mut S, n: NumberValue) -> Result<(), S::Error> {
    match n {
        NumberValue::Integer(i) => {
            let mut buf = itoa::Buffer::new();
            out.write_bytes(buf.format(i).as_bytes())
        }
        NumberValue::Float(f) => {
            if !f.is_finite() {
                // serde_json emits non-finite floats as `null` to keep
                // output valid JSON. Match that.
                return out.write_bytes(b"null");
            }
            let mut buf = ryu::Buffer::new();
            out.write_bytes(buf.format_finite(f).as_bytes())
        }
    }
}

// ---- Compact emit (no whitespace) ------------------------------------------------

fn write_data_value<S: JsonSink>(out: &mut S, v: &DataValue<'_>) -> Result<(), S::Error> {
    match *v {
        DataValue::Null => out.write_bytes(b"null"),
        DataValue::Bool(true) => out.write_bytes(b"true"),
        DataValue::Bool(false) => out.write_bytes(b"false"),
        DataValue::Number(n) => write_number(out, n),
        DataValue::String(s) => write_escaped_str(out, s),
        DataValue::Array(items) => {
            out.write_byte(b'[')?;
            let mut first = true;
            for item in items {
                if !first {
                    out.write_byte(b',')?;
                }
                first = false;
                write_data_value(out, item)?;
            }
            out.write_byte(b']')
        }
        DataValue::Object(pairs) => {
            out.write_byte(b'{')?;
            let mut first = true;
            for (k, v) in pairs {
                if !first {
                    out.write_byte(b',')?;
                }
                first = false;
                write_escaped_str(out, k)?;
                out.write_byte(b':')?;
                write_data_value(out, v)?;
            }
            out.write_byte(b'}')
        }
        #[cfg(feature = "datetime")]
        DataValue::DateTime(d) => write_escaped_str(out, &d.to_iso_string()),
        #[cfg(feature = "datetime")]
        DataValue::Duration(d) => write_escaped_str(out, &d.to_string()),
    }
}

fn write_owned_value<S: JsonSink>(out: &mut S, v: &OwnedDataValue) -> Result<(), S::Error> {
    match v {
        OwnedDataValue::Null => out.write_bytes(b"null"),
        OwnedDataValue::Bool(true) => out.write_bytes(b"true"),
        OwnedDataValue::Bool(false) => out.write_bytes(b"false"),
        OwnedDataValue::Number(n) => write_number(out, *n),
        OwnedDataValue::String(s) => write_escaped_str(out, s),
        OwnedDataValue::Array(items) => {
            out.write_byte(b'[')?;
            let mut first = true;
            for item in items {
                if !first {
                    out.write_byte(b',')?;
                }
                first = false;
                write_owned_value(out, item)?;
            }
            out.write_byte(b']')
        }
        OwnedDataValue::Object(pairs) => {
            out.write_byte(b'{')?;
            let mut first = true;
            for (k, v) in pairs {
                if !first {
                    out.write_byte(b',')?;
                }
                first = false;
                write_escaped_str(out, k)?;
                out.write_byte(b':')?;
                write_owned_value(out, v)?;
            }
            out.write_byte(b'}')
        }
        #[cfg(feature = "datetime")]
        OwnedDataValue::DateTime(d) => write_escaped_str(out, &d.to_iso_string()),
        #[cfg(feature = "datetime")]
        OwnedDataValue::Duration(d) => write_escaped_str(out, &d.to_string()),
    }
}

// ---- Pretty emit (two-space indent, matches serde_json::to_string_pretty) -------

#[inline]
fn write_indent<S: JsonSink>(out: &mut S, depth: usize) -> Result<(), S::Error> {
    // Two spaces per level. Keep a reasonably long literal so most depths
    // need a single write.
    const SPACES: &[u8; 64] = b"                                                                ";
    let mut remaining = depth * 2;
    while remaining > 0 {
        let chunk = remaining.min(SPACES.len());
        out.write_bytes(&SPACES[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

fn write_data_value_pretty<S: JsonSink>(
    out: &mut S,
    v: &DataValue<'_>,
    depth: usize,
) -> Result<(), S::Error> {
    match *v {
        DataValue::Null => out.write_bytes(b"null"),
        DataValue::Bool(true) => out.write_bytes(b"true"),
        DataValue::Bool(false) => out.write_bytes(b"false"),
        DataValue::Number(n) => write_number(out, n),
        DataValue::String(s) => write_escaped_str(out, s),
        DataValue::Array(items) => {
            if items.is_empty() {
                return out.write_bytes(b"[]");
            }
            out.write_byte(b'[')?;
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.write_byte(b',')?;
                }
                out.write_byte(b'\n')?;
                write_indent(out, depth + 1)?;
                write_data_value_pretty(out, item, depth + 1)?;
            }
            out.write_byte(b'\n')?;
            write_indent(out, depth)?;
            out.write_byte(b']')
        }
        DataValue::Object(pairs) => {
            if pairs.is_empty() {
                return out.write_bytes(b"{}");
            }
            out.write_byte(b'{')?;
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.write_byte(b',')?;
                }
                out.write_byte(b'\n')?;
                write_indent(out, depth + 1)?;
                write_escaped_str(out, k)?;
                out.write_bytes(b": ")?;
                write_data_value_pretty(out, v, depth + 1)?;
            }
            out.write_byte(b'\n')?;
            write_indent(out, depth)?;
            out.write_byte(b'}')
        }
        #[cfg(feature = "datetime")]
        DataValue::DateTime(d) => write_escaped_str(out, &d.to_iso_string()),
        #[cfg(feature = "datetime")]
        DataValue::Duration(d) => write_escaped_str(out, &d.to_string()),
    }
}

fn write_owned_value_pretty<S: JsonSink>(
    out: &mut S,
    v: &OwnedDataValue,
    depth: usize,
) -> Result<(), S::Error> {
    match v {
        OwnedDataValue::Null => out.write_bytes(b"null"),
        OwnedDataValue::Bool(true) => out.write_bytes(b"true"),
        OwnedDataValue::Bool(false) => out.write_bytes(b"false"),
        OwnedDataValue::Number(n) => write_number(out, *n),
        OwnedDataValue::String(s) => write_escaped_str(out, s),
        OwnedDataValue::Array(items) => {
            if items.is_empty() {
                return out.write_bytes(b"[]");
            }
            out.write_byte(b'[')?;
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.write_byte(b',')?;
                }
                out.write_byte(b'\n')?;
                write_indent(out, depth + 1)?;
                write_owned_value_pretty(out, item, depth + 1)?;
            }
            out.write_byte(b'\n')?;
            write_indent(out, depth)?;
            out.write_byte(b']')
        }
        OwnedDataValue::Object(pairs) => {
            if pairs.is_empty() {
                return out.write_bytes(b"{}");
            }
            out.write_byte(b'{')?;
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.write_byte(b',')?;
                }
                out.write_byte(b'\n')?;
                write_indent(out, depth + 1)?;
                write_escaped_str(out, k)?;
                out.write_bytes(b": ")?;
                write_owned_value_pretty(out, v, depth + 1)?;
            }
            out.write_byte(b'\n')?;
            write_indent(out, depth)?;
            out.write_byte(b'}')
        }
        #[cfg(feature = "datetime")]
        OwnedDataValue::DateTime(d) => write_escaped_str(out, &d.to_iso_string()),
        #[cfg(feature = "datetime")]
        OwnedDataValue::Duration(d) => write_escaped_str(out, &d.to_string()),
    }
}

// ---- Public API on DataValue ---------------------------------------------------

impl DataValue<'_> {
    /// Append the compact JSON encoding of this value to `out`. Useful when
    /// you want to amortize allocation across many values into a shared buffer.
    /// For one-shot string conversion, use the [`fmt::Display`] impl
    /// (`v.to_string()` / `format!("{v}")` / `println!("{v}")`).
    pub fn write_json_into(&self, out: &mut Vec<u8>) {
        let _ = write_data_value(out, self);
    }

    /// Pretty-print wrapper. `format!("{}", v.pretty())` produces the same
    /// two-space-indented layout as `serde_json::to_string_pretty`.
    ///
    /// ```
    /// use bumpalo::Bump;
    /// use datavalue_rs::DataValue;
    ///
    /// let arena = Bump::new();
    /// let v = DataValue::from_str(r#"{"a":1}"#, &arena).unwrap();
    /// assert_eq!(v.pretty().to_string(), "{\n  \"a\": 1\n}");
    /// ```
    pub fn pretty(&self) -> Pretty<'_, DataValue<'_>> {
        Pretty(self)
    }

    /// Append the pretty JSON encoding of this value to `out`.
    pub fn write_json_pretty_into(&self, out: &mut Vec<u8>) {
        let _ = write_data_value_pretty(out, self, 0);
    }
}

impl OwnedDataValue {
    /// Append the compact JSON encoding of this value to `out`. See
    /// [`DataValue::write_json_into`]; this is the owned-side mirror.
    pub fn write_json_into(&self, out: &mut Vec<u8>) {
        let _ = write_owned_value(out, self);
    }

    /// Pretty-print wrapper; see [`DataValue::pretty`].
    ///
    /// ```
    /// use datavalue_rs::OwnedDataValue;
    ///
    /// let v: OwnedDataValue = r#"{"a":1}"#.parse().unwrap();
    /// assert_eq!(v.pretty().to_string(), "{\n  \"a\": 1\n}");
    /// ```
    pub fn pretty(&self) -> Pretty<'_, OwnedDataValue> {
        Pretty(self)
    }

    /// Append the pretty JSON encoding of this value to `out`.
    pub fn write_json_pretty_into(&self, out: &mut Vec<u8>) {
        let _ = write_owned_value_pretty(out, self, 0);
    }
}

// ---- Display + Pretty wrapper ---------------------------------------------------

/// Wrapper produced by [`DataValue::pretty`] / [`OwnedDataValue::pretty`] that
/// renders the value as indented JSON via `Display`.
pub struct Pretty<'b, T: ?Sized>(&'b T);

impl fmt::Display for DataValue<'_> {
    /// Compact JSON. Same shape as `serde_json::to_string`.
    ///
    /// ```
    /// use bumpalo::Bump;
    /// use datavalue_rs::DataValue;
    ///
    /// let arena = Bump::new();
    /// let v = DataValue::from_str(r#"{"a":[1,2.5,"hi"]}"#, &arena).unwrap();
    /// assert_eq!(v.to_string(), r#"{"a":[1,2.5,"hi"]}"#);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_data_value(&mut FormatterSink(f), self)
    }
}

impl fmt::Display for OwnedDataValue {
    /// Compact JSON. Same shape as `serde_json::to_string`.
    ///
    /// ```
    /// use datavalue_rs::OwnedDataValue;
    ///
    /// let v: OwnedDataValue = r#"{"a":[1,2.5,"hi"]}"#.parse().unwrap();
    /// assert_eq!(v.to_string(), r#"{"a":[1,2.5,"hi"]}"#);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_owned_value(&mut FormatterSink(f), self)
    }
}

impl fmt::Display for Pretty<'_, DataValue<'_>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_data_value_pretty(&mut FormatterSink(f), self.0, 0)
    }
}

impl fmt::Display for Pretty<'_, OwnedDataValue> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_owned_value_pretty(&mut FormatterSink(f), self.0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    fn round_trip(s: &str) -> String {
        let arena = Bump::new();
        let v = DataValue::from_str(s, &arena).unwrap();
        v.to_string()
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
        assert_eq!(v.to_string(), "\"\\u0001\"");
    }

    #[test]
    fn nested_round_trip_matches_serde_json() {
        let input = r#"{"a":[1,2,{"b":"hi\n","c":null,"d":true}],"e":-3.5,"f":[],"g":{}}"#;
        let arena = Bump::new();
        let v = DataValue::from_str(input, &arena).unwrap();
        let ours = v.to_string();
        let serde: serde_json::Value = serde_json::from_str(input).unwrap();
        let theirs = serde_json::to_string(&serde).unwrap();
        assert_eq!(ours, theirs);
    }

    #[test]
    fn long_string_swar_path() {
        let arena = Bump::new();
        let s = format!("\"{}\"", "x".repeat(200));
        let v = DataValue::from_str(&s, &arena).unwrap();
        assert_eq!(v.to_string(), s);
    }

    #[test]
    fn non_finite_floats_render_as_null() {
        let v = DataValue::from_f64(f64::NAN);
        assert_eq!(v.to_string(), "null");
        let v = DataValue::from_f64(f64::INFINITY);
        assert_eq!(v.to_string(), "null");
    }

    #[test]
    fn owned_round_trip() {
        let v: OwnedDataValue = r#"{"name":"alice","age":30}"#.parse().unwrap();
        let serde: serde_json::Value = serde_json::from_str(&v.to_string()).unwrap();
        assert_eq!(serde["name"], "alice");
        assert_eq!(serde["age"], 30);
    }

    #[test]
    fn write_json_into_buffer() {
        let arena = Bump::new();
        let v = DataValue::from_str(r#"[1,2,3]"#, &arena).unwrap();
        let mut buf = Vec::new();
        v.write_json_into(&mut buf);
        assert_eq!(buf, b"[1,2,3]");
    }

    #[test]
    fn pretty_matches_serde_json_pretty() {
        let input = r#"{"a":[1,2,{"b":"hi","c":null}],"e":-3.5,"f":[],"g":{}}"#;
        let arena = Bump::new();
        let v = DataValue::from_str(input, &arena).unwrap();
        let ours = v.pretty().to_string();
        let serde: serde_json::Value = serde_json::from_str(input).unwrap();
        let theirs = serde_json::to_string_pretty(&serde).unwrap();
        assert_eq!(ours, theirs);
    }

    #[test]
    fn pretty_owned_matches_serde_json_pretty() {
        let input = r#"{"a":[1,2,{"b":"hi","c":null}],"e":-3.5,"f":[],"g":{}}"#;
        let v: OwnedDataValue = input.parse().unwrap();
        let serde: serde_json::Value = serde_json::from_str(input).unwrap();
        assert_eq!(
            v.pretty().to_string(),
            serde_json::to_string_pretty(&serde).unwrap()
        );
    }

    #[test]
    fn pretty_empty_collections_inline() {
        let arena = Bump::new();
        let v = DataValue::from_str(r#"{"a":[],"b":{}}"#, &arena).unwrap();
        assert_eq!(v.pretty().to_string(), "{\n  \"a\": [],\n  \"b\": {}\n}");
    }

    #[test]
    fn pretty_deep_indent_beyond_64_spaces() {
        // 35 levels deep -> 70 spaces of indent on the leaf line. Exercises
        // the chunked SPACES write loop.
        let arena = Bump::new();
        let mut s = String::new();
        for _ in 0..35 {
            s.push('[');
        }
        s.push('1');
        for _ in 0..35 {
            s.push(']');
        }
        let v = DataValue::from_str(&s, &arena).unwrap();
        let ours = v.pretty().to_string();
        let serde: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(ours, serde_json::to_string_pretty(&serde).unwrap());
    }
}
