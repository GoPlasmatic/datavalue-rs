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

mod swar {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGHS: u64 = 0x8080_8080_8080_8080;

    #[inline(always)]
    fn mask(w: u64) -> u64 {
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
}
