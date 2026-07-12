//! gRPC service skeleton (tonic, feature-gated).
//!
//! Compiled when the `grpc` feature is enabled. Generated proto code lives in OUT_DIR.

/// HippmemService placeholder struct (replaced by tonic trait impl once proto is compiled).
#[cfg(feature = "grpc")]
pub struct HippmemService;

#[cfg(feature = "grpc")]
impl HippmemService {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "grpc")]
impl Default for HippmemService {
    fn default() -> Self {
        Self::new()
    }
}
