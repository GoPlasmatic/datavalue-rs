//! Bump-allocated JSON parser. See [`DataValue::from_str`] for the entry point.
//!
//! Strategy:
//! - Single linear scan over the input bytes.
//! - Strings without escape sequences are borrowed directly from the input
//!   (zero-copy). Strings with escapes are unescaped into the arena.
//! - Arrays/objects are accumulated in `bumpalo::collections::Vec` then
//!   frozen into `&[..]` slices via `into_bump_slice`.
//! - Numbers parse on the integer fast path (i64) and only fall back to f64
//!   when a decimal point or exponent is present (or i64 overflows).

use core::fmt;

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

use crate::number::NumberValue;
use crate::value::DataValue;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorKind {
    UnexpectedEof,
    UnexpectedByte(u8),
    InvalidEscape,
    InvalidUnicodeEscape,
    InvalidNumber,
    TrailingData,
    DepthLimitExceeded,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json parse error at byte {}: ", self.position)?;
        match self.kind {
            ParseErrorKind::UnexpectedEof => write!(f, "unexpected end of input"),
            ParseErrorKind::UnexpectedByte(b) => {
                write!(f, "unexpected byte 0x{:02x} ({:?})", b, b as char)
            }
            ParseErrorKind::InvalidEscape => write!(f, "invalid string escape"),
            ParseErrorKind::InvalidUnicodeEscape => write!(f, "invalid \\u escape"),
            ParseErrorKind::InvalidNumber => write!(f, "invalid number literal"),
            ParseErrorKind::TrailingData => write!(f, "unexpected data after JSON value"),
            ParseErrorKind::DepthLimitExceeded => write!(f, "nesting depth limit exceeded"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Soft cap on nested array/object depth. Keeps the stack usage bounded so
/// pathological input can't blow the recursive descent stack. 256 is well
/// past anything legitimate JSON would produce.
const MAX_DEPTH: u16 = 256;

impl<'a> DataValue<'a> {
    /// Parse a JSON document into a [`DataValue`] tree allocated in `arena`.
    ///
    /// Strings without escape sequences are borrowed directly from `input`
    /// (the returned tree's lifetime is the shorter of `input` and `arena`).
    pub fn from_str(input: &'a str, arena: &'a Bump) -> Result<DataValue<'a>, ParseError> {
        let mut p = Parser {
            bytes: input.as_bytes(),
            input,
            pos: 0,
            arena,
        };
        p.skip_ws();
        let value = p.parse_value(0)?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(p.err(ParseErrorKind::TrailingData));
        }
        Ok(value)
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    input: &'a str,
    pos: usize,
    arena: &'a Bump,
}

impl<'a> Parser<'a> {
    #[inline(always)]
    fn err(&self, kind: ParseErrorKind) -> ParseError {
        ParseError {
            kind,
            position: self.pos,
        }
    }

    #[inline(always)]
    fn peek(&self) -> Result<u8, ParseError> {
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| self.err(ParseErrorKind::UnexpectedEof))
    }

    #[inline(always)]
    fn bump(&mut self) -> Result<u8, ParseError> {
        let b = self.peek()?;
        self.pos += 1;
        Ok(b)
    }

    #[inline(always)]
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn parse_value(&mut self, depth: u16) -> Result<DataValue<'a>, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.err(ParseErrorKind::DepthLimitExceeded));
        }
        self.skip_ws();
        let b = self.peek()?;
        match b {
            b'"' => self.parse_string().map(DataValue::String),
            b'{' => self.parse_object(depth),
            b'[' => self.parse_array(depth),
            b't' | b'f' => self.parse_bool(),
            b'n' => self.parse_null(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(self.err(ParseErrorKind::UnexpectedByte(other))),
        }
    }

    fn parse_null(&mut self) -> Result<DataValue<'a>, ParseError> {
        if self.bytes.get(self.pos..self.pos + 4) == Some(b"null") {
            self.pos += 4;
            Ok(DataValue::Null)
        } else {
            Err(self.err(ParseErrorKind::UnexpectedByte(self.bytes[self.pos])))
        }
    }

    fn parse_bool(&mut self) -> Result<DataValue<'a>, ParseError> {
        if self.bytes.get(self.pos..self.pos + 4) == Some(b"true") {
            self.pos += 4;
            Ok(DataValue::Bool(true))
        } else if self.bytes.get(self.pos..self.pos + 5) == Some(b"false") {
            self.pos += 5;
            Ok(DataValue::Bool(false))
        } else {
            Err(self.err(ParseErrorKind::UnexpectedByte(self.bytes[self.pos])))
        }
    }

    fn parse_number(&mut self) -> Result<DataValue<'a>, ParseError> {
        let start = self.pos;
        let mut is_float = false;

        if self.bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        // Integer part.
        match self.peek()? {
            b'0' => {
                self.pos += 1;
            }
            b'1'..=b'9' => {
                self.pos += 1;
                while let Some(&c) = self.bytes.get(self.pos) {
                    if c.is_ascii_digit() {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
            }
            _ => return Err(self.err(ParseErrorKind::InvalidNumber)),
        }
        // Fraction.
        if let Some(&b'.') = self.bytes.get(self.pos) {
            is_float = true;
            self.pos += 1;
            let frac_start = self.pos;
            while let Some(&c) = self.bytes.get(self.pos) {
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == frac_start {
                return Err(self.err(ParseErrorKind::InvalidNumber));
            }
        }
        // Exponent.
        if let Some(&c) = self.bytes.get(self.pos)
            && (c == b'e' || c == b'E')
        {
            is_float = true;
            self.pos += 1;
            if let Some(&s) = self.bytes.get(self.pos)
                && (s == b'+' || s == b'-')
            {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while let Some(&d) = self.bytes.get(self.pos) {
                if d.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == exp_start {
                return Err(self.err(ParseErrorKind::InvalidNumber));
            }
        }

        let slice = &self.input[start..self.pos];
        if !is_float {
            // Integer fast path.
            if let Ok(i) = slice.parse::<i64>() {
                return Ok(DataValue::Number(NumberValue::Integer(i)));
            }
            // Fall through to f64 on i64 overflow.
        }
        match slice.parse::<f64>() {
            Ok(f) => Ok(DataValue::Number(NumberValue::Float(f))),
            Err(_) => Err(ParseError {
                kind: ParseErrorKind::InvalidNumber,
                position: start,
            }),
        }
    }

    /// Parse a `"..."` string and return the resolved &str. Borrowed from
    /// the input when there are no escape sequences; otherwise unescaped
    /// into the arena.
    fn parse_string(&mut self) -> Result<&'a str, ParseError> {
        // Already at the opening quote.
        debug_assert_eq!(self.bytes[self.pos], b'"');
        self.pos += 1;
        let start = self.pos;

        // Fast path: scan for terminator or escape without copying.
        loop {
            let b = match self.bytes.get(self.pos) {
                Some(&b) => b,
                None => return Err(self.err(ParseErrorKind::UnexpectedEof)),
            };
            match b {
                b'"' => {
                    let s = &self.input[start..self.pos];
                    self.pos += 1;
                    return Ok(s);
                }
                b'\\' => {
                    // Switch to slow path: copy what we have so far, then
                    // resolve escapes one at a time.
                    return self.parse_string_with_escapes(start);
                }
                0..=0x1F => {
                    return Err(self.err(ParseErrorKind::UnexpectedByte(b)));
                }
                _ => self.pos += 1,
            }
        }
    }

    fn parse_string_with_escapes(&mut self, start: usize) -> Result<&'a str, ParseError> {
        let mut out: BumpVec<u8> = BumpVec::with_capacity_in(self.pos - start + 16, self.arena);
        out.extend_from_slice(&self.bytes[start..self.pos]);

        loop {
            let b = match self.bytes.get(self.pos) {
                Some(&b) => b,
                None => return Err(self.err(ParseErrorKind::UnexpectedEof)),
            };
            match b {
                b'"' => {
                    self.pos += 1;
                    let slice = out.into_bump_slice();
                    // The input is &str (already valid UTF-8) and our
                    // unescape path only ever produces valid UTF-8 byte
                    // sequences, so this is sound.
                    return Ok(unsafe { core::str::from_utf8_unchecked(slice) });
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self.bump()?;
                    match esc {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            // Handle surrogate pairs.
                            let ch = if (0xD800..=0xDBFF).contains(&code) {
                                if self.bytes.get(self.pos) != Some(&b'\\')
                                    || self.bytes.get(self.pos + 1) != Some(&b'u')
                                {
                                    return Err(self.err(ParseErrorKind::InvalidUnicodeEscape));
                                }
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(self.err(ParseErrorKind::InvalidUnicodeEscape));
                                }
                                let scalar = 0x10000
                                    + (((code - 0xD800) as u32) << 10)
                                    + ((low - 0xDC00) as u32);
                                char::from_u32(scalar)
                                    .ok_or_else(|| self.err(ParseErrorKind::InvalidUnicodeEscape))?
                            } else if (0xDC00..=0xDFFF).contains(&code) {
                                return Err(self.err(ParseErrorKind::InvalidUnicodeEscape));
                            } else {
                                char::from_u32(code as u32)
                                    .ok_or_else(|| self.err(ParseErrorKind::InvalidUnicodeEscape))?
                            };
                            let mut buf = [0u8; 4];
                            let s = ch.encode_utf8(&mut buf);
                            out.extend_from_slice(s.as_bytes());
                        }
                        _ => return Err(self.err(ParseErrorKind::InvalidEscape)),
                    }
                }
                0..=0x1F => return Err(self.err(ParseErrorKind::UnexpectedByte(b))),
                _ => {
                    out.push(b);
                    self.pos += 1;
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, ParseError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(self.err(ParseErrorKind::InvalidUnicodeEscape));
        }
        let mut v: u16 = 0;
        for _ in 0..4 {
            let b = self.bytes[self.pos];
            let d = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(self.err(ParseErrorKind::InvalidUnicodeEscape)),
            } as u16;
            v = (v << 4) | d;
            self.pos += 1;
        }
        Ok(v)
    }

    fn parse_array(&mut self, depth: u16) -> Result<DataValue<'a>, ParseError> {
        debug_assert_eq!(self.bytes[self.pos], b'[');
        self.pos += 1;
        self.skip_ws();
        let mut items: BumpVec<DataValue<'a>> = BumpVec::new_in(self.arena);
        if let Some(&b']') = self.bytes.get(self.pos) {
            self.pos += 1;
            return Ok(DataValue::Array(items.into_bump_slice()));
        }
        loop {
            let v = self.parse_value(depth + 1)?;
            items.push(v);
            self.skip_ws();
            match self.bump()? {
                b',' => self.skip_ws(),
                b']' => return Ok(DataValue::Array(items.into_bump_slice())),
                other => return Err(self.err(ParseErrorKind::UnexpectedByte(other))),
            }
        }
    }

    fn parse_object(&mut self, depth: u16) -> Result<DataValue<'a>, ParseError> {
        debug_assert_eq!(self.bytes[self.pos], b'{');
        self.pos += 1;
        self.skip_ws();
        let mut pairs: BumpVec<(&'a str, DataValue<'a>)> = BumpVec::new_in(self.arena);
        if let Some(&b'}') = self.bytes.get(self.pos) {
            self.pos += 1;
            return Ok(DataValue::Object(pairs.into_bump_slice()));
        }
        loop {
            self.skip_ws();
            if self.peek()? != b'"' {
                return Err(self.err(ParseErrorKind::UnexpectedByte(self.bytes[self.pos])));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bump()? != b':' {
                return Err(self.err(ParseErrorKind::UnexpectedByte(self.bytes[self.pos - 1])));
            }
            self.skip_ws();
            let value = self.parse_value(depth + 1)?;
            pairs.push((key, value));
            self.skip_ws();
            match self.bump()? {
                b',' => continue,
                b'}' => return Ok(DataValue::Object(pairs.into_bump_slice())),
                other => return Err(self.err(ParseErrorKind::UnexpectedByte(other))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> DataValue<'_> {
        let arena = Box::leak(Box::new(Bump::new()));
        DataValue::from_str(s, arena).expect("parse")
    }

    #[test]
    fn primitives() {
        assert!(parse("null").is_null());
        assert_eq!(parse("true").as_bool(), Some(true));
        assert_eq!(parse("false").as_bool(), Some(false));
        assert_eq!(parse("0").as_i64(), Some(0));
        assert_eq!(parse("-7").as_i64(), Some(-7));
        assert_eq!(parse("3.5").as_f64(), Some(3.5));
        assert_eq!(parse("1e3").as_f64(), Some(1000.0));
        assert_eq!(parse(r#""hello""#).as_str(), Some("hello"));
    }

    #[test]
    fn integer_overflow_falls_to_float() {
        let v = parse("123456789012345678901234567890");
        assert!(v.is_f64());
    }

    #[test]
    fn empty_collections() {
        assert_eq!(parse("[]").len(), Some(0));
        assert_eq!(parse("{}").len(), Some(0));
    }

    #[test]
    fn arrays_and_objects() {
        let v = parse(r#"{"a":[1,2,3],"b":{"c":true}}"#);
        assert_eq!(v["a"][0].as_i64(), Some(1));
        assert_eq!(v["a"][2].as_i64(), Some(3));
        assert_eq!(v["b"]["c"].as_bool(), Some(true));
    }

    #[test]
    fn string_escapes() {
        assert_eq!(parse(r#""a\nb""#).as_str(), Some("a\nb"));
        assert_eq!(parse(r#""a\\b""#).as_str(), Some("a\\b"));
        assert_eq!(parse(r#""é""#).as_str(), Some("é"));
        // Surrogate pair for U+1F600 😀
        assert_eq!(parse(r#""😀""#).as_str(), Some("😀"));
    }

    #[test]
    fn whitespace_tolerant() {
        let v = parse(" {\n \"a\" :\t1 ,\n \"b\":2\n} ");
        assert_eq!(v["a"].as_i64(), Some(1));
        assert_eq!(v["b"].as_i64(), Some(2));
    }

    #[test]
    fn rejects_trailing_data() {
        let arena = Bump::new();
        assert!(DataValue::from_str("1 2", &arena).is_err());
    }

    #[test]
    fn rejects_bad_escape() {
        let arena = Bump::new();
        assert!(DataValue::from_str(r#""\q""#, &arena).is_err());
    }

    #[test]
    fn deep_nesting_under_limit_ok() {
        let n = 200;
        let s = "[".repeat(n) + &"]".repeat(n);
        let arena = Bump::new();
        assert!(DataValue::from_str(&s, &arena).is_ok());
    }

    #[test]
    fn deep_nesting_over_limit_errors() {
        let n = 1000;
        let s = "[".repeat(n) + &"]".repeat(n);
        let arena = Bump::new();
        assert!(DataValue::from_str(&s, &arena).is_err());
    }
}
