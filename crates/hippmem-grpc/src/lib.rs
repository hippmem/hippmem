//! HIPPMEM gRPC thin wrapper: proto + tonic service (ADR-011/021).

#[cfg(feature = "grpc")]
pub mod service;

#[cfg(test)]
mod tests {
    #[test]
    fn grpc_feature_gated() {
        // Compiles successfully when the grpc feature is disabled (default)
    }
}
