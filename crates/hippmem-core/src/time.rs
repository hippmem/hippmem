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
