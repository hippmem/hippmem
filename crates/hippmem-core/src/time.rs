//! Timestamp newtype (Timestamp) and the injectable Clock trait.
//!
//! Corresponds to ADR-007 and 02#0. Constitution §4.3: all "now" inside the library MUST be obtained via the Clock trait,
//! and **MUST NOT** call `SystemTime::now()` directly.

use serde::{Deserialize, Serialize};

/// Unix millisecond timestamp (UTC). An i64 newtype. See ADR-007.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(pub i64);

impl Timestamp {
    /// Extracts the inner i64 value (Unix milliseconds).
    pub fn as_i64(&self) -> i64 {
        self.0
    }

    /// Constructs from Unix milliseconds.
    pub fn from_millis(ms: i64) -> Self {
        Self(ms)
    }
}

// ── Clock trait ──

/// Injectable clock: all "now" inside the library is obtained through this trait.
///
/// See constitution §4.3 (test reproducibility), ADR-007.
pub trait Clock {
    /// Returns the current UTC timestamp (Unix milliseconds).
    fn now(&self) -> Timestamp;
}

/// System clock: uses `std::time::SystemTime::now()` to obtain real time.
///
/// **For application layer and factory methods only**; library logic MUST obtain time through the `Clock` trait.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        Timestamp(ms)
    }
}

/// Fixed clock: returns a fixed Timestamp, used for testing.
pub struct FixedClock {
    timestamp: Timestamp,
}

impl FixedClock {
    /// Creates a fixed clock that always returns the given timestamp.
    pub fn new(timestamp: Timestamp) -> Self {
        Self { timestamp }
    }

    /// Sets a new fixed timestamp.
    pub fn set(&mut self, timestamp: Timestamp) {
        self.timestamp = timestamp;
    }
}

impl Clock for FixedClock {
    fn now(&self) -> Timestamp {
        self.timestamp
    }
}

/// Days since 1970-01-01 for a civil date (proleptic Gregorian).
/// Howard Hinnant's `days_from_civil` algorithm.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy =
        (153 * (if month > 2 { month - 3 } else { month + 9 }) as i64 + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Reverse of [`days_from_civil`]: civil date for a day count since epoch.
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}
