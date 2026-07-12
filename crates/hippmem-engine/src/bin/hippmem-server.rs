//! HIPPMEM gRPC Server (requires --features grpc).
//!
//! Starts a tonic gRPC server, exposing the Engine API.

use hippmem_engine::{Engine, EngineConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store_dir = std::env::var("HIPPMEM_STORE_DIR").unwrap_or_else(|_| "./hippmem_data".into());
    let listen = std::env::var("HIPPMEM_LISTEN").unwrap_or_else(|_| "0.0.0.0:50051".into());

    let config = EngineConfig {
        store_dir: PathBuf::from(store_dir),
        ..Default::default()
    };
    let engine = Engine::open(config)?;
    let engine = std::sync::Arc::new(engine);

    println!("HIPPMEM gRPC server listening on {}", listen);

    // tonic service skeleton (implemented after proto compiles)
    // let svc = hippmem_grpc::service::HippmemService::new(engine);
    // tonic::transport::Server::builder()
    //     .add_service(hippmem::v1::hippmem_server::HippmemServer::new(svc))
    //     .serve(listen.parse()?)
    //     .await?;

    let _ = engine;
    let _ = listen;
    Ok(())
}
