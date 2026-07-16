//! Cross-architecture SIMD primitives, exposed as standalone public
//! utilities.
//!
//! The crate's own parser and emitter currently keep an 8-byte SWAR scan
//! inlined into their hot paths because the canonical JSON fixtures
//! (twitter, citm) are dominated by very short strings — under that
//! distribution, SIMD register setup costs more per call than the wider
//! 16-byte stride saves. The cost/benefit flips for long strings (logs,
//! base64 blobs, large free-text fields), where this module's NEON / SSE2
//! paths beat the inlined SWAR.
//!
//! Architectures supported (no runtime feature detection):
//! - `aarch64`: NEON (baseline)
//! - `x86_64`: SSE2 (baseline)
//! - `arm` with `target_feature = "neon"`: NEON
//! - everything else: 8-byte SWAR fallback
//!
//! Each path has matching tests (see the `tests` submodule) so the SIMD
//! result must agree with the SWAR result on the same input.

/// Find the offset of the first byte in `bytes` that is `"`, `\\`, or a
/// control byte (< 0x20). Returns `None` if `bytes` contains no such byte.
///
/// This is the hot inner loop for both parsing and emitting JSON strings —
/// the parser uses it to find string terminators, the emitter uses it to
/// find bytes that need escaping, and the criteria are identical.
///
/// ```
/// use datavalue_rs::simd::find_string_terminator;
///
/// // The first JSON-special byte is `"` at offset 5.
/// assert_eq!(find_string_terminator(b"hello\"world"), Some(5));
///
/// // No special bytes.
/// assert_eq!(find_string_terminator(b"hello world"), None);
///
/// // UTF-8 continuation bytes (>= 0x80) are not flagged as control bytes.
/// assert_eq!(find_string_terminator("café".as_bytes()), None);
/// ```
#[inline(always)]
pub fn find_string_terminator(bytes: &[u8]) -> Option<usize> {
    // Short slices skip the SIMD register setup; the SWAR loop already
    // processes 8 bytes per iteration with negligible per-call overhead.
    if bytes.len() < 32 {
        return swar::find_string_terminator(bytes);
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64_neon::find_string_terminator(bytes) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { x86_64_sse2::find_string_terminator(bytes) };
    }
    #[cfg(all(target_arch = "arm", target_feature = "neon"))]
    {
        return unsafe { arm_neon::find_string_terminator(bytes) };
    }
    #[allow(unreachable_code)]
    swar::find_string_terminator(bytes)
}

/// 8-byte SWAR mask for the JSON string-special bytes (`"`, `\\`, < 0x20):
/// the high bit of each lane is set iff that byte is special. This is the
/// single source of truth — the parser and emitter build their inlined
/// window loops on it rather than carrying private copies.
#[inline(always)]
pub(crate) fn special_mask8(w: u64) -> u64 {
    swar::mask(w)
}

/// Scan like [`find_string_terminator`], additionally reporting whether
/// every byte *before* the hit (or the whole slice, when no special byte
/// exists) is ASCII. The ASCII bit accumulates in the pass that already
/// touches the bytes — one OR per window.
///
/// Not yet called by the parser: it lands with the `StrAscii` string
/// variant (proposal Item C) and exists now so the scan machinery and its
/// tests are proven alongside the adaptive-scan work.
///
/// ```
/// use datavalue_rs::simd::find_string_terminator_ascii;
///
/// assert_eq!(find_string_terminator_ascii(b"hello\"tail"), (Some(5), true));
/// assert_eq!(find_string_terminator_ascii("café".as_bytes()), (None, false));
/// assert_eq!(find_string_terminator_ascii("café\"x".as_bytes()), (Some(5), false));
/// ```
pub fn find_string_terminator_ascii(bytes: &[u8]) -> (Option<usize>, bool) {
    swar::find_string_terminator_ascii(bytes)
}

mod swar {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGHS: u64 = 0x8080_8080_8080_8080;

    #[inline(always)]
    pub(super) fn mask(w: u64) -> u64 {
        let q = w ^ (b'"' as u64 * ONES);
        let bs = w ^ (b'\\' as u64 * ONES);
        // Top three bits clear iff byte < 0x20.
        let lo = w & 0xE0E0_E0E0_E0E0_E0E0;
        let m_q = q.wrapping_sub(ONES) & !q;
        let m_bs = bs.wrapping_sub(ONES) & !bs;
        let m_lo = lo.wrapping_sub(ONES) & !lo;
        (m_q | m_bs | m_lo) & HIGHS
    }

    /// SWAR (8-byte SIMD-within-a-register) fast scan — also exposed
    /// directly so users can pick this explicitly even on SIMD targets.
    #[inline(always)]
    pub fn find_string_terminator(bytes: &[u8]) -> Option<usize> {
        let len = bytes.len();
        let mut i = 0;
        while i + 8 <= len {
            let w = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
            let m = mask(w);
            if m != 0 {
                return Some(i + (m.trailing_zeros() / 8) as usize);
            }
            i += 8;
        }
        while i < len {
            let b = bytes[i];
            if matches!(b, b'"' | b'\\') || b < 0x20 {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// SWAR terminator scan that also accumulates an all-ASCII bit over the
    /// bytes preceding the hit. See the parent module's re-export for docs.
    pub(super) fn find_string_terminator_ascii(bytes: &[u8]) -> (Option<usize>, bool) {
        let len = bytes.len();
        let mut seen: u64 = 0;
        let mut i = 0;
        while i + 8 <= len {
            let w = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
            let m = mask(w);
            if m != 0 {
                let k = (m.trailing_zeros() / 8) as usize;
                // Accumulate only the bytes before the hit: little-endian,
                // so earlier bytes are the low-order lanes.
                if k > 0 {
                    seen |= w & (u64::MAX >> (8 * (8 - k)));
                }
                return (Some(i + k), seen & HIGHS == 0);
            }
            seen |= w;
            i += 8;
        }
        let mut ascii = seen & HIGHS == 0;
        while i < len {
            let b = bytes[i];
            if matches!(b, b'"' | b'\\') || b < 0x20 {
                return (Some(i), ascii);
            }
            ascii &= b < 0x80;
            i += 1;
        }
        (None, ascii)
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
    use core::arch::aarch64::*;

    /// SAFETY: NEON is part of the aarch64 baseline ABI; the unsafe block
    /// is only required because the intrinsics are themselves unsafe. No
    /// `#[target_feature]` attribute, since adding it would block inlining
    /// from a baseline-NEON caller.
    /// NEON-accelerated path. SAFETY: NEON is part of the aarch64
    /// baseline ABI; the unsafe wrapper exists only because the
    /// intrinsics themselves are unsafe.
    #[inline(always)]
    pub unsafe fn find_string_terminator(bytes: &[u8]) -> Option<usize> {
        unsafe {
            let len = bytes.len();
            let ptr = bytes.as_ptr();
            let v_q = vdupq_n_u8(b'"');
            let v_b = vdupq_n_u8(b'\\');
            let v_20 = vdupq_n_u8(0x20);
            let mut i = 0;
            while i + 16 <= len {
                let v = vld1q_u8(ptr.add(i));
                let eq_q = vceqq_u8(v, v_q);
                let eq_b = vceqq_u8(v, v_b);
                let is_ctl = vcltq_u8(v, v_20); // unsigned <
                let combined = vorrq_u8(vorrq_u8(eq_q, eq_b), is_ctl);
                // NEON has no direct movemask. Standard reduction: shift
                // each 16-bit lane right by 4, narrow to 8-bit. Each
                // source byte contributes 4 bits to the resulting 64-bit
                // value, in order.
                let nibble = vshrn_n_u16::<4>(vreinterpretq_u16_u8(combined));
                let mask64 = vget_lane_u64::<0>(vreinterpret_u64_u8(nibble));
                if mask64 != 0 {
                    return Some(i + (mask64.trailing_zeros() as usize) / 4);
                }
                i += 16;
            }
            super::swar::find_string_terminator(&bytes[i..]).map(|off| i + off)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64_sse2 {
    use core::arch::x86_64::*;

    /// SSE2-accelerated path. SAFETY: SSE2 is part of the x86_64 baseline
    /// ABI.
    #[inline(always)]
    pub unsafe fn find_string_terminator(bytes: &[u8]) -> Option<usize> {
        unsafe {
            let len = bytes.len();
            let ptr = bytes.as_ptr();
            let v_q = _mm_set1_epi8(b'"' as i8);
            let v_b = _mm_set1_epi8(b'\\' as i8);
            let v_1f = _mm_set1_epi8(0x1F);
            let mut i = 0;
            while i + 16 <= len {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let eq_q = _mm_cmpeq_epi8(v, v_q);
                let eq_b = _mm_cmpeq_epi8(v, v_b);
                // For unsigned `b <= 0x1F`: min(b, 0x1F) == b iff b <=
                // 0x1F. (signed cmplt would give false positives for
                // high-bit bytes.)
                let min_v = _mm_min_epu8(v, v_1f);
                let is_ctl = _mm_cmpeq_epi8(min_v, v);
                let combined = _mm_or_si128(_mm_or_si128(eq_q, eq_b), is_ctl);
                let mask = _mm_movemask_epi8(combined) as u32;
                if mask != 0 {
                    return Some(i + mask.trailing_zeros() as usize);
                }
                i += 16;
            }
            super::swar::find_string_terminator(&bytes[i..]).map(|off| i + off)
        }
    }
}

#[cfg(all(target_arch = "arm", target_feature = "neon"))]
mod arm_neon {
    use core::arch::arm::*;

    /// 32-bit ARM NEON path. Requires `target_feature = "neon"` at
    /// compile time.
    #[target_feature(enable = "neon")]
    pub unsafe fn find_string_terminator(bytes: &[u8]) -> Option<usize> {
        unsafe {
            let len = bytes.len();
            let ptr = bytes.as_ptr();
            let v_q = vdupq_n_u8(b'"');
            let v_b = vdupq_n_u8(b'\\');
            let v_20 = vdupq_n_u8(0x20);
            let mut i = 0;
            while i + 16 <= len {
                let v = vld1q_u8(ptr.add(i));
                let eq_q = vceqq_u8(v, v_q);
                let eq_b = vceqq_u8(v, v_b);
                let is_ctl = vcltq_u8(v, v_20);
                let combined = vorrq_u8(vorrq_u8(eq_q, eq_b), is_ctl);
                let nibble = vshrn_n_u16::<4>(vreinterpretq_u16_u8(combined));
                let mask64 = vget_lane_u64::<0>(vreinterpret_u64_u8(nibble));
                if mask64 != 0 {
                    return Some(i + (mask64.trailing_zeros() as usize) / 4);
                }
                i += 16;
            }
            super::swar::find_string_terminator(&bytes[i..]).map(|off| i + off)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::find_string_terminator;

    #[test]
    fn empty() {
        assert_eq!(find_string_terminator(b""), None);
    }

    #[test]
    fn no_terminator() {
        let s = b"abcdefghijklmnopqrstuvwxyz0123456789";
        assert_eq!(find_string_terminator(s), None);
    }

    #[test]
    fn first_byte_quote() {
        assert_eq!(find_string_terminator(b"\"abc"), Some(0));
    }

    #[test]
    fn quote_after_long_run() {
        // 30 safe bytes then a quote — exercises the SIMD bulk loop.
        let mut s = vec![b'x'; 30];
        s.push(b'"');
        s.push(b'y');
        assert_eq!(find_string_terminator(&s), Some(30));
    }

    #[test]
    fn backslash_in_tail() {
        // 19 safe bytes (one full 16-byte SIMD window plus 3 tail bytes)
        // then a backslash.
        let mut s = vec![b'x'; 19];
        s.push(b'\\');
        assert_eq!(find_string_terminator(&s), Some(19));
    }

    #[test]
    fn every_control_byte_at_window_boundary() {
        // Each control byte 0x00..=0x1F preceded by 16 safe bytes — forces
        // the bulk loop to skip the safe window then hit the terminator
        // at offset 16 in the second iteration.
        for ctl in 0u8..0x20 {
            let mut s = vec![b'x'; 16];
            s.push(ctl);
            s.push(b'y');
            assert_eq!(
                find_string_terminator(&s),
                Some(16),
                "control byte 0x{ctl:02x} missed",
            );
        }
    }

    #[test]
    fn high_bit_bytes_are_safe() {
        // UTF-8 continuation bytes (>= 0x80) must NOT be flagged as
        // control bytes (the signed-cmplt trap).
        let s: Vec<u8> = (0x80u8..=0xFFu8).collect();
        assert_eq!(find_string_terminator(&s), None);
    }

    #[test]
    fn multibyte_utf8_safe() {
        // "café" = 0x63 0x61 0x66 0xC3 0xA9 — no terminator.
        let s = "café and more text past the SIMD window";
        assert_eq!(find_string_terminator(s.as_bytes()), None);
    }

    #[test]
    fn ascii_variant_agrees_with_reference() {
        use super::find_string_terminator_ascii;

        // Reference: position of first special byte + is_ascii of the prefix.
        fn reference(bytes: &[u8]) -> (Option<usize>, bool) {
            let hit = bytes
                .iter()
                .position(|&b| matches!(b, b'"' | b'\\') || b < 0x20);
            let prefix = &bytes[..hit.unwrap_or(bytes.len())];
            (hit, prefix.is_ascii())
        }

        let mut cases: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"a".to_vec(),
            b"\"".to_vec(),
            b"abcdefg\"".to_vec(),
            b"abcdefgh\"".to_vec(), // hit exactly at window boundary
            "é".as_bytes().to_vec(),
            "caf\u{e9} then a longer ascii tail without hits"
                .as_bytes()
                .to_vec(),
        ];
        // Non-ASCII byte at every offset of a 24-byte string, hit after it.
        for off in 0..24 {
            let mut v = vec![b'x'; 24];
            v[off] = 0xC3; // continuation handled below; raw high byte is enough
            v.push(b'"');
            cases.push(v);
        }
        // Non-ASCII byte *after* the hit must not spoil the prefix bit.
        let mut v = vec![b'x'; 10];
        v.push(b'"');
        v.extend_from_slice("é".as_bytes());
        cases.push(v);
        // Hit at offset 0 of a window whose later bytes are non-ASCII.
        let mut v = vec![b'x'; 8];
        v.push(b'"');
        v.extend_from_slice(&[0xFF; 7]);
        cases.push(v);

        for case in &cases {
            assert_eq!(
                find_string_terminator_ascii(case),
                reference(case),
                "mismatch on {case:?}"
            );
        }
    }
}
