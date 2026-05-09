//! `DataDateTime` / `DataDuration` — chrono-backed value types behind the
//! `datetime` feature. Mirrors the shape exposed by `datalogic-rs` so the
//! two crates can share a common JSON value tree.
//!
//! These live as inline variants on [`crate::DataValue`] (`DateTime(..)`
//! / `Duration(..)`). Both types are `Copy` so [`crate::DataValue`] stays
//! `Copy` regardless of feature selection.
//!
//! The JSON parser does not produce datetime nodes — JSON has no native
//! representation. Consumers that want to upgrade `String` → `DateTime`
//! call [`DataDateTime::parse`] explicitly (typically inside an
//! operator/coercion boundary in a downstream crate).

use core::fmt;

use chrono::{DateTime, Duration, NaiveDateTime, Utc};

/// A UTC datetime plus the original timezone offset, so round-tripping
/// preserves what the input string said. The offset is informational —
/// equality / ordering compare the underlying UTC instant.
#[derive(Debug, Clone, Copy)]
pub struct DataDateTime {
    pub dt: DateTime<Utc>,
    /// Original timezone offset in seconds. `Some(0)` for `Z` input,
    /// `None` for naive input (no timezone in the source string).
    pub original_offset: Option<i32>,
}

/// A signed duration. Wraps `chrono::Duration`.
#[derive(Debug, Clone, Copy)]
pub struct DataDuration(pub Duration);

// ---- Equality / ordering ----
//
// Compare instants only; the original_offset is metadata. This matches
// what every other datetime-aware system does ("2024-01-01T00:00:00Z" ==
// "2024-01-01T01:00:00+01:00").

impl PartialEq for DataDateTime {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.dt == other.dt
    }
}
impl Eq for DataDateTime {}
impl PartialOrd for DataDateTime {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DataDateTime {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.dt.cmp(&other.dt)
    }
}

impl PartialEq for DataDuration {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for DataDuration {}
impl PartialOrd for DataDuration {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DataDuration {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

// ---- DataDateTime ----

impl DataDateTime {
    /// Saturate datetime to max/min bounds on overflow.
    #[inline]
    fn saturate(dt: Option<DateTime<Utc>>, is_positive: bool) -> DateTime<Utc> {
        dt.unwrap_or(if is_positive {
            DateTime::<Utc>::MAX_UTC
        } else {
            DateTime::<Utc>::MIN_UTC
        })
    }

    /// Parse RFC 3339 / ISO 8601. Falls back to `%Y-%m-%dT%H:%M:%S` (naive,
    /// assumed UTC) so input from systems that elide the timezone still
    /// round-trips.
    pub fn parse(s: &str) -> Option<Self> {
        // Fast path: exact "YYYY-MM-DDTHH:MM:SSZ" (20 bytes, UTC) — what
        // `now`-style ISO output produces.
        let bytes = s.as_bytes();
        if bytes.len() == 20
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[10] == b'T'
            && bytes[13] == b':'
            && bytes[16] == b':'
            && bytes[19] == b'Z'
        {
            if let Some(dt) = Self::parse_utc_fast(bytes) {
                return Some(dt);
            }
        }

        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            let offset = dt.offset().local_minus_utc();
            return Some(DataDateTime {
                dt: dt.with_timezone(&Utc),
                original_offset: Some(offset),
            });
        }

        if let Ok(naive) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
            return Some(DataDateTime {
                dt: DateTime::from_naive_utc_and_offset(naive, Utc),
                original_offset: None,
            });
        }

        None
    }

    #[inline]
    fn parse_utc_fast(b: &[u8]) -> Option<Self> {
        let year = parse_4digits(b, 0)? as i32;
        let month = parse_2digits(b, 5)?;
        let day = parse_2digits(b, 8)?;
        let hour = parse_2digits(b, 11)?;
        let min = parse_2digits(b, 14)?;
        let sec = parse_2digits(b, 17)?;
        let date = chrono::NaiveDate::from_ymd_opt(year, month, day)?;
        let time = chrono::NaiveTime::from_hms_opt(hour, min, sec)?;
        let naive = NaiveDateTime::new(date, time);
        Some(DataDateTime {
            dt: DateTime::from_naive_utc_and_offset(naive, Utc),
            original_offset: Some(0),
        })
    }

    /// Parse with an explicit `chrono::format` specifier. Tries datetime
    /// first, then date-only (with midnight time).
    pub fn parse_with_format(s: &str, format: &str) -> Option<Self> {
        if let Ok(naive) = NaiveDateTime::parse_from_str(s, format) {
            return Some(DataDateTime {
                dt: DateTime::from_naive_utc_and_offset(naive, Utc),
                original_offset: None,
            });
        }
        if let Ok(date) = chrono::NaiveDate::parse_from_str(s, format) {
            let dt = date.and_hms_opt(0, 0, 0)?;
            return Some(DataDateTime {
                dt: DateTime::from_naive_utc_and_offset(dt, Utc),
                original_offset: None,
            });
        }
        None
    }

    /// `format == "z"` is special: returns the original tz offset like
    /// `"+0530"` (or `"+0000"` if unknown). Otherwise delegates to chrono.
    pub fn format(&self, format: &str) -> String {
        if format == "z" {
            if let Some(offset_secs) = self.original_offset {
                let hours = offset_secs / 3600;
                let minutes = (offset_secs % 3600).abs() / 60;
                return format!("{:+03}{:02}", hours, minutes);
            }
            return "+0000".to_string();
        }
        self.dt.format(format).to_string()
    }

    /// `YYYY-MM-DDTHH:MM:SSZ` — the canonical wire format.
    pub fn to_iso_string(&self) -> String {
        format_utc_iso_secs(&self.dt)
    }

    pub fn add_duration(&self, duration: &DataDuration) -> DataDateTime {
        let dt = Self::saturate(
            self.dt.checked_add_signed(duration.0),
            duration.0.num_seconds() > 0,
        );
        DataDateTime {
            dt,
            original_offset: self.original_offset,
        }
    }

    pub fn sub_duration(&self, duration: &DataDuration) -> DataDateTime {
        let dt = Self::saturate(
            self.dt.checked_sub_signed(duration.0),
            duration.0.num_seconds() < 0,
        );
        DataDateTime {
            dt,
            original_offset: self.original_offset,
        }
    }

    pub fn diff(&self, other: &DataDateTime) -> DataDuration {
        DataDuration(self.dt - other.dt)
    }

    pub fn diff_in_unit(&self, other: &DataDateTime, unit: &str) -> f64 {
        let duration = self.dt - other.dt;
        match unit {
            "days" => duration.num_days() as f64,
            "hours" => duration.num_hours() as f64,
            "minutes" => duration.num_minutes() as f64,
            "seconds" => duration.num_seconds() as f64,
            "milliseconds" => duration.num_milliseconds() as f64,
            _ => 0.0,
        }
    }
}

impl fmt::Display for DataDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_iso_string())
    }
}

// ---- DataDuration ----

impl DataDuration {
    fn saturate(seconds: f64) -> DataDuration {
        if !seconds.is_finite() || seconds > i64::MAX as f64 / 1000.0 {
            DataDuration(Duration::MAX)
        } else if seconds < i64::MIN as f64 / 1000.0 {
            DataDuration(Duration::MIN)
        } else {
            Duration::try_seconds(seconds as i64)
                .map(DataDuration)
                .unwrap_or(DataDuration(Duration::MAX))
        }
    }

    /// Parse `"1d:2h:3m:4s"` / `"1d2h30m"` / `"45s"` style durations.
    /// Returns `None` if no unit suffix is present at all.
    pub fn parse(s: &str) -> Option<Self> {
        let mut days = 0i64;
        let mut hours = 0i64;
        let mut minutes = 0i64;
        let mut seconds = 0i64;

        if s.contains(':') {
            for part in s.split(':') {
                if let Some(stripped) = part.strip_suffix('d') {
                    days = stripped.parse().ok()?;
                } else if let Some(stripped) = part.strip_suffix('h') {
                    hours = stripped.parse().ok()?;
                } else if let Some(stripped) = part.strip_suffix('m') {
                    minutes = stripped.parse().ok()?;
                } else if let Some(stripped) = part.strip_suffix('s') {
                    seconds = stripped.parse().ok()?;
                }
            }
        } else {
            let mut current = String::new();
            for ch in s.chars() {
                if ch.is_ascii_digit() {
                    current.push(ch);
                } else {
                    let n: i64 = current.parse().ok()?;
                    current.clear();
                    match ch {
                        'd' => days = n,
                        'h' => hours = n,
                        'm' => minutes = n,
                        's' => seconds = n,
                        _ => return None,
                    }
                }
            }
        }

        if days == 0
            && hours == 0
            && minutes == 0
            && seconds == 0
            && !s.contains(['d', 'h', 'm', 's'])
        {
            return None;
        }

        let total = days
            .checked_mul(86_400)?
            .checked_add(hours.checked_mul(3_600)?)?
            .checked_add(minutes.checked_mul(60)?)?
            .checked_add(seconds)?;

        if !(i64::MIN / 1000..=i64::MAX / 1000).contains(&total) {
            return Some(DataDuration(if total > 0 {
                Duration::MAX
            } else {
                Duration::MIN
            }));
        }
        Duration::try_seconds(total).map(DataDuration)
    }

    pub fn multiply(&self, factor: f64) -> DataDuration {
        let result = self.0.num_seconds() as f64 * factor;
        if !result.is_finite() {
            DataDuration(self.0)
        } else {
            Self::saturate(result)
        }
    }

    pub fn divide(&self, divisor: f64) -> DataDuration {
        if divisor == 0.0 || divisor.abs() < f64::EPSILON {
            return DataDuration(Duration::MAX);
        }
        let result = self.0.num_seconds() as f64 / divisor;
        if !result.is_finite() {
            DataDuration(self.0)
        } else {
            Self::saturate(result)
        }
    }

    pub fn add(&self, other: &DataDuration) -> DataDuration {
        self.0
            .checked_add(&other.0)
            .map(DataDuration)
            .unwrap_or_else(|| {
                if self.0.num_seconds() > 0 || other.0.num_seconds() > 0 {
                    DataDuration(Duration::MAX)
                } else {
                    DataDuration(Duration::MIN)
                }
            })
    }

    pub fn sub(&self, other: &DataDuration) -> DataDuration {
        self.0
            .checked_sub(&other.0)
            .map(DataDuration)
            .unwrap_or_else(|| {
                if self.0.num_seconds() > other.0.num_seconds() {
                    DataDuration(Duration::MAX)
                } else {
                    DataDuration(Duration::MIN)
                }
            })
    }
}

impl fmt::Display for DataDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.0.num_seconds();
        let days = total / 86_400;
        let hours = (total % 86_400) / 3_600;
        let minutes = (total % 3_600) / 60;
        let seconds = total % 60;
        write!(f, "{}d:{}h:{}m:{}s", days, hours, minutes, seconds)
    }
}

// ---- Helpers ----

#[inline]
fn format_utc_iso_secs(dt: &DateTime<Utc>) -> String {
    use chrono::{Datelike, Timelike};
    let year = dt.year();
    if !(0..=9999).contains(&year) {
        return dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    }
    let month = dt.month() as u8;
    let day = dt.day() as u8;
    let hour = dt.hour() as u8;
    let minute = dt.minute() as u8;
    let second = dt.second() as u8;

    let y = year as u16;
    let mut buf = [0u8; 20];
    buf[0] = b'0' + ((y / 1000) % 10) as u8;
    buf[1] = b'0' + ((y / 100) % 10) as u8;
    buf[2] = b'0' + ((y / 10) % 10) as u8;
    buf[3] = b'0' + (y % 10) as u8;
    buf[4] = b'-';
    buf[5] = b'0' + month / 10;
    buf[6] = b'0' + month % 10;
    buf[7] = b'-';
    buf[8] = b'0' + day / 10;
    buf[9] = b'0' + day % 10;
    buf[10] = b'T';
    buf[11] = b'0' + hour / 10;
    buf[12] = b'0' + hour % 10;
    buf[13] = b':';
    buf[14] = b'0' + minute / 10;
    buf[15] = b'0' + minute % 10;
    buf[16] = b':';
    buf[17] = b'0' + second / 10;
    buf[18] = b'0' + second % 10;
    buf[19] = b'Z';

    // SAFETY: every byte written is ASCII.
    unsafe { String::from_utf8_unchecked(buf.to_vec()) }
}

#[inline(always)]
fn parse_2digits(b: &[u8], offset: usize) -> Option<u32> {
    let d0 = b[offset].wrapping_sub(b'0');
    let d1 = b[offset + 1].wrapping_sub(b'0');
    if d0 > 9 || d1 > 9 {
        return None;
    }
    Some(d0 as u32 * 10 + d1 as u32)
}

#[inline(always)]
fn parse_4digits(b: &[u8], offset: usize) -> Option<u32> {
    let d0 = b[offset].wrapping_sub(b'0');
    let d1 = b[offset + 1].wrapping_sub(b'0');
    let d2 = b[offset + 2].wrapping_sub(b'0');
    let d3 = b[offset + 3].wrapping_sub(b'0');
    if d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9 {
        return None;
    }
    Some(d0 as u32 * 1000 + d1 as u32 * 100 + d2 as u32 * 10 + d3 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iso_z_fast_path() {
        let d = DataDateTime::parse("2024-01-15T12:30:45Z").unwrap();
        assert_eq!(d.original_offset, Some(0));
        assert_eq!(d.to_iso_string(), "2024-01-15T12:30:45Z");
    }

    #[test]
    fn parse_rfc3339_with_offset_preserves_offset() {
        let d = DataDateTime::parse("2024-01-15T12:30:45+05:30").unwrap();
        assert_eq!(d.original_offset, Some(5 * 3600 + 30 * 60));
        assert_eq!(d.format("z"), "+0530");
    }

    #[test]
    fn parse_naive_assumes_utc() {
        let d = DataDateTime::parse("2024-01-15T12:30:45").unwrap();
        assert_eq!(d.original_offset, None);
        assert_eq!(d.format("z"), "+0000");
    }

    #[test]
    fn duration_parse_colon_form() {
        let d = DataDuration::parse("1d:2h:3m:4s").unwrap();
        assert_eq!(d.0.num_seconds(), 86_400 + 7_200 + 180 + 4);
    }

    #[test]
    fn duration_parse_compact_form() {
        let d = DataDuration::parse("1d2h30m").unwrap();
        assert_eq!(d.0.num_seconds(), 86_400 + 7_200 + 1_800);
    }

    #[test]
    fn duration_display_round_trips_through_parse() {
        let d = DataDuration::parse("3d:5h:7m:11s").unwrap();
        assert_eq!(d.to_string(), "3d:5h:7m:11s");
        let r = DataDuration::parse(&d.to_string()).unwrap();
        assert_eq!(d, r);
    }

    #[test]
    fn datetime_arith() {
        let a = DataDateTime::parse("2024-01-15T00:00:00Z").unwrap();
        let dur = DataDuration::parse("1d").unwrap();
        let b = a.add_duration(&dur);
        assert_eq!(b.to_iso_string(), "2024-01-16T00:00:00Z");
        assert_eq!(b.sub_duration(&dur), a);
    }

    #[test]
    fn datetime_eq_compares_instants() {
        let a = DataDateTime::parse("2024-01-15T12:00:00Z").unwrap();
        let b = DataDateTime::parse("2024-01-15T17:30:00+05:30").unwrap();
        // Same UTC instant, different originating tz.
        assert_eq!(a, b);
    }
}
