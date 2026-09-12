//! Standard base64 (RFC 4648 §4: `+` `/` alphabet, `=` padding, no line
//! breaks) for the `tensor` wire format.
//!
//! Hand-rolled so the base `tensor` feature stays dependency-free. The
//! encoder streams through [`JsonSink`] in fixed chunks, so a large buffer
//! never materialises as one `String` on the native emit path; the decoder
//! is strict (no whitespace, padding only at the end, length a multiple of
//! four) and writes into a caller-provided buffer so the arena decoder can
//! land bytes directly in aligned memory.

#[cfg(any(feature = "serde", test))]
use core::fmt;

use crate::emit::JsonSink;

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Reverse table. `0xFF` marks bytes outside the alphabet; `=` is also
/// `0xFF` here and is handled by position in [`decode_into`].
const DECODE: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut i = 0;
    while i < 64 {
        t[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    t
};

/// Encoded length of `n` input bytes (always a multiple of four).
#[cfg(test)]
#[inline]
pub(crate) fn encoded_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

/// Four output characters for one 24-bit group.
#[inline]
fn quad(n: u32) -> [u8; 4] {
    [
        ALPHABET[((n >> 18) & 63) as usize],
        ALPHABET[((n >> 12) & 63) as usize],
        ALPHABET[((n >> 6) & 63) as usize],
        ALPHABET[(n & 63) as usize],
    ]
}

/// Encode `input` into `out`, which must hold at least
/// `encoded_len(input.len())` bytes. Returns the number of bytes written.
fn encode_block(input: &[u8], out: &mut [u8]) -> usize {
    let mut o = 0;
    let (chunks, rem) = input.as_chunks::<3>();
    for c in chunks {
        let n = ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32);
        out[o..o + 4].copy_from_slice(&quad(n));
        o += 4;
    }
    // A short tail is a zero-padded group whose unused characters become `=`.
    if !rem.is_empty() {
        let mut last = [0u8; 3];
        last[..rem.len()].copy_from_slice(rem);
        let n = ((last[0] as u32) << 16) | ((last[1] as u32) << 8) | (last[2] as u32);
        let mut q = quad(n);
        if rem.len() == 1 {
            q[2] = b'=';
        }
        q[3] = b'=';
        out[o..o + 4].copy_from_slice(&q);
        o += 4;
    }
    o
}

/// Encode `data` 3072 input bytes (4096 output bytes) at a time, handing
/// each encoded chunk to `sink`. The output is pure base64 alphabet —
/// ASCII with no JSON-special bytes — so a chunk is a complete valid UTF-8
/// run and needs no escape scan.
fn encode_chunks<E>(data: &[u8], mut sink: impl FnMut(&[u8]) -> Result<(), E>) -> Result<(), E> {
    const IN: usize = 3072;
    const OUT: usize = 4096;
    let mut buf = [0u8; OUT];
    for chunk in data.chunks(IN) {
        let n = encode_block(chunk, &mut buf);
        sink(&buf[..n])?;
    }
    Ok(())
}

/// Stream the encoding of `data` into a [`JsonSink`] (the native emitter).
pub(crate) fn encode_into<S: JsonSink>(out: &mut S, data: &[u8]) -> Result<(), S::Error> {
    encode_chunks(data, |b| out.write_bytes(b))
}

/// `Display` adapter so serde's `collect_str` can stream the encoding too.
#[cfg(any(feature = "serde", test))]
pub(crate) struct Base64<'a>(pub(crate) &'a [u8]);

#[cfg(any(feature = "serde", test))]
impl fmt::Display for Base64<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: every byte written is from the ASCII alphabet or `=`.
        encode_chunks(self.0, |b| {
            f.write_str(unsafe { core::str::from_utf8_unchecked(b) })
        })
    }
}

/// One-shot encode to a `String`.
#[cfg(test)]
pub(crate) fn encode_string(data: &[u8]) -> String {
    Base64(data).to_string()
}

/// Byte length of the decoded form, or `None` if `input` cannot be valid
/// base64 by length and padding alone.
pub(crate) fn decoded_len(input: &[u8]) -> Option<usize> {
    if !input.len().is_multiple_of(4) {
        return None;
    }
    if input.is_empty() {
        return Some(0);
    }
    let pad = if input.ends_with(b"==") {
        2
    } else if input[input.len() - 1] == b'=' {
        1
    } else {
        0
    };
    Some(input.len() / 4 * 3 - pad)
}

/// Decode `input` into `out`, which must be exactly `decoded_len(input)`
/// bytes long. Returns `false` on any alphabet or padding violation (`out`
/// may then be partially written). Trailing bits under the padding are not
/// required to be zero, matching the common lenient decoders.
pub(crate) fn decode_into(input: &[u8], out: &mut [u8]) -> bool {
    debug_assert_eq!(decoded_len(input), Some(out.len()));
    if input.is_empty() {
        return true;
    }
    let (body, last) = input.split_at(input.len() - 4);
    let mut o = 0;
    for q in body.as_chunks::<4>().0 {
        let (a, b, c, d) = (
            DECODE[q[0] as usize],
            DECODE[q[1] as usize],
            DECODE[q[2] as usize],
            DECODE[q[3] as usize],
        );
        // Valid sextets have the top two bits clear; 0xFF has them set.
        if (a | b | c | d) & 0xC0 != 0 {
            return false;
        }
        let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | (d as u32);
        out[o] = (n >> 16) as u8;
        out[o + 1] = (n >> 8) as u8;
        out[o + 2] = n as u8;
        o += 3;
    }

    let (a, b) = (DECODE[last[0] as usize], DECODE[last[1] as usize]);
    if (a | b) & 0xC0 != 0 {
        return false;
    }
    let (c, d, produced) = match (last[2], last[3]) {
        (b'=', b'=') => (0, 0, 1),
        (b'=', _) => return false,
        (x, b'=') => (DECODE[x as usize], 0, 2),
        (x, y) => (DECODE[x as usize], DECODE[y as usize], 3),
    };
    if (c | d) & 0xC0 != 0 {
        return false;
    }
    let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | (d as u32);
    let tail = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
    out[o..o + produced].copy_from_slice(&tail[..produced]);
    o += produced;
    o == out.len()
}

/// Convenience one-shot decode.
#[cfg(test)]
pub(crate) fn decode(input: &str) -> Option<Vec<u8>> {
    let n = decoded_len(input.as_bytes())?;
    let mut out = vec![0u8; n];
    decode_into(input.as_bytes(), &mut out).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc4648_vectors() {
        let cases: &[(&[u8], &str)] = &[
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
        ];
        for (raw, enc) in cases {
            assert_eq!(encode_string(raw), *enc);
            assert_eq!(decode(enc).as_deref(), Some(*raw));
        }
    }

    #[test]
    fn round_trip_every_length_and_across_chunk_boundary() {
        for len in (0..=64).chain([3071, 3072, 3073, 6144, 10_000]) {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let s = encode_string(&data);
            assert_eq!(s.len(), encoded_len(len));
            let mut streamed = Vec::new();
            encode_into(&mut streamed, &data).unwrap();
            assert_eq!(
                streamed,
                s.as_bytes(),
                "streamed encode differs at len {len}"
            );
            assert_eq!(decode(&s).unwrap(), data, "decode differs at len {len}");
        }
    }

    #[test]
    fn strict_decoder_rejects_malformed_input() {
        for bad in [
            "abc", "ab=c", "====", "a b c d", "abcd=", "Zm9v\n", "Zm9v!", "=abc",
        ] {
            assert!(decode(bad).is_none(), "accepted {bad:?}");
        }
    }
}
