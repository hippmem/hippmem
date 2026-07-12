//! Bounded score UnitScore: 0.0..=1.0, clamped at construction. See 02#0.

use serde::{Deserialize, Serialize};

/// Bounded score in 0.0..=1.0 (strength/confidence/importance).
///
/// **Invariant**: the inner f32 is always within `[0.0, 1.0]`.
/// `UnitScore::new(x)` clamps out-of-range values and maps NaN to zero.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct UnitScore(f32);

impl UnitScore {
    /// Constructs a UnitScore, automatically clamped to [0.0, 1.0]. NaN maps to zero.
    pub fn new(value: f32) -> Self {
        if value.is_nan() {
            Self(0.0)
        } else {
            Self(value.clamp(0.0, 1.0))
        }
    }

    /// Returns the inner f32 value (always in [0.0, 1.0]).
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Default for UnitScore {
    fn default() -> Self {
        Self(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_midpoint() {
        assert!((UnitScore::default().value() - 0.5).abs() < f32::EPSILON);
    }
}
