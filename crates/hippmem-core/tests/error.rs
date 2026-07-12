//! acceptance test: basic error types and Result alias

use hippmem_core::error::{CoreError, CoreResult};

// ── Error construction ──

#[test]
fn create_schema_too_new() {
    let err = CoreError::SchemaTooNew {
        found: 99,
        current: 1,
    };
    assert_eq!(err.found_version(), 99);
    assert_eq!(err.current_version(), 1);
}

#[test]
fn create_validation_error() {
    let err = CoreError::Validation("value out of range".into());
    let msg = format!("{}", err);
    assert!(
        msg.contains("value out of range"),
        "error message should contain the original text: {}",
        msg
    );
}

#[test]
fn create_serialization_error() {
    let err = CoreError::Serialization("encoding failed".into());
    let msg = format!("{}", err);
    assert!(
        msg.contains("encoding failed"),
        "error message should contain the original text: {}",
        msg
    );
}

// ── SchemaTooNew field access ──

#[test]
fn schema_too_new_fields() {
    let err = CoreError::SchemaTooNew {
        found: 5,
        current: 2,
    };
    assert_eq!(err.found_version(), 5);
    assert_eq!(err.current_version(), 2);
}

// ── Display / Debug impls ──

#[test]
fn error_display_is_readable() {
    let err = CoreError::SchemaTooNew {
        found: 10,
        current: 3,
    };
    let display = format!("{}", err);
    assert!(!display.is_empty(), "Display output should not be empty");
    assert!(
        display.contains("10") || display.contains("3"),
        "Display should contain the version number: {}",
        display
    );
}

#[test]
fn error_debug_is_verbose() {
    let err = CoreError::SchemaTooNew {
        found: 10,
        current: 3,
    };
    let debug = format!("{:?}", err);
    assert!(!debug.is_empty(), "Debug output should not be empty");
}

// ── CoreResult alias ──

#[test]
fn core_result_ok() {
    let result: CoreResult<i32> = Ok(42);
    assert!(result.is_ok());
    assert_eq!(result, Ok(42));
}

#[test]
fn core_result_err() {
    let result: CoreResult<i32> = Err(CoreError::Validation("test error".into()));
    assert!(result.is_err());
}

// ── Error is matchable (pattern matching) ──

#[test]
fn error_pattern_matching() {
    let err = CoreError::SchemaTooNew {
        found: 7,
        current: 1,
    };
    match err {
        CoreError::SchemaTooNew { found, current } => {
            assert_eq!(found, 7);
            assert_eq!(current, 1);
        }
        _ => panic!("should match SchemaTooNew"),
    }
}
