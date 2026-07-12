# gRPC Guide

> HIPPMEM provides a gRPC interface, callable from Python, Go, Node.js, and other languages.
> The gRPC feature is gated behind the `grpc` feature and is not compiled by default.

---

## Starting the gRPC Server

```bash
# Build (requires the grpc feature)
cargo build --bin hippmem-server

# Start (listens on 0.0.0.0:50051 by default)
./target/debug/hippmem-server

# Custom storage directory and port
HIPPMEM_STORE_DIR=./my_data HIPPMEM_LISTEN=0.0.0.0:50052 ./target/debug/hippmem-server
```

Or via the `serve` subcommand of the CLI (also requires `--features grpc`):
```bash
cargo run --bin hippmem -- -s ./my_data serve
```

---

## Proto Overview

The proto file is at `crates/hippmem-grpc/proto/hippmem.proto`.

```protobuf
service Hippmem {
  rpc Write(WriteRequest) returns (WriteResponse);
  rpc Retrieve(RetrieveRequest) returns (RetrieveResponse);
  rpc Explain(ExplainRequest) returns (ExplainResponse);
  rpc Consolidate(ConsolidateRequest) returns (ConsolidateResponse);
  rpc Inspect(InspectRequest) returns (InspectResponse);
  rpc Feedback(FeedbackRequest) returns (FeedbackResponse);
}
```

**All 6 RPCs are Unary (not streaming).**

---

## Python Client

### Install Dependencies

```bash
pip install grpcio grpcio-tools
```

### Generate Code

```bash
git clone https://github.com/hippmem/hippmem.git
cd hippmem

python -m grpc_tools.protoc \
  -I crates/hippmem-grpc/proto \
  --python_out=./py_client \
  --grpc_python_out=./py_client \
  crates/hippmem-grpc/proto/hippmem.proto
```

### Full Example

```python
"""hippmem_client.py — HIPPMEM gRPC Python client"""
import grpc
import hippmem_pb2
import hippmem_pb2_grpc


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = hippmem_pb2_grpc.HippmemStub(channel)

    # ── Write ──
    resp = stub.Write(hippmem_pb2.WriteRequest(
        content="The user is a software engineer who prefers Rust and uses redb for embedded storage.",
        content_type="Preference",
        importance_hint=0.8,
    ))
    print(f"write ok: memory_id={resp.memory_id} stage={resp.stage}")

    # ── Retrieve ──
    resp = stub.Retrieve(hippmem_pb2.RetrieveRequest(
        query="What language does the user prefer?",
        top_k=5,
        mode="Balanced",
    ))
    for item in resp.results:
        print(f"[{item.score:.3f}] {item.content[:60]} dims={list(item.dimensions)}")

    # ── Explain ──
    resp = stub.Explain(hippmem_pb2.ExplainRequest(memory_id=1))
    print(f"importance: {resp.importance:.3f} linked: {resp.link_count}")

    # ── Consolidate ──
    resp = stub.Consolidate(hippmem_pb2.ConsolidateRequest(scope="Incremental"))
    print(f"consolidate: decayed {resp.edges_decayed} edges, elapsed {resp.elapsed_ms}ms")

    # ── Inspect ──
    resp = stub.Inspect(hippmem_pb2.InspectRequest(query="StoreStats"))
    print(f"total memories: {resp.store_stats.memory_count}")

    channel.close()


if __name__ == "__main__":
    main()
```

---

## Go Client

### Install Dependencies

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

### Generate Code

```bash
protoc \
  -I crates/hippmem-grpc/proto \
  --go_out=./go_client \
  --go-grpc_out=./go_client \
  crates/hippmem-grpc/proto/hippmem.proto
```

### Full Example

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    pb "go_client/hippmem/v1"
)

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()

    conn, err := grpc.NewClient("localhost:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()

    client := pb.NewHippmemClient(conn)

    // Write
    writeResp, err := client.Write(ctx, &pb.WriteRequest{
        Content:        "The user prefers Rust for backend development and values clean error handling patterns.",
        ContentType:    strPtr("Preference"),
        ImportanceHint: floatPtr(0.7),
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("write ok: memory_id=%d\n", writeResp.MemoryId)

    // Retrieve
    retrieveResp, err := client.Retrieve(ctx, &pb.RetrieveRequest{
        Query: "What language does the user prefer?",
        TopK:  5,
        Mode:  "Balanced",
    })
    if err != nil {
        log.Fatal(err)
    }
    for _, item := range retrieveResp.Results {
        fmt.Printf("[%.3f] %s dims=%v\n",
            item.Score, item.Content[:60], item.Dimensions)
    }
}

func strPtr(s string) *string { return &s }
func floatPtr(f float32) *float32 { return &f }
```

---

## Node.js Client

### Install Dependencies

```bash
npm install @grpc/grpc-js @grpc/proto-loader
```

### Full Example

```javascript
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");

const PROTO_PATH = "../crates/hippmem-grpc/proto/hippmem.proto";
const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: Number,
  enums: String,
  defaults: true,
  oneofs: true,
});

const hippmem = grpc.loadPackageDefinition(packageDefinition).hippmem.v1;
const client = new hippmem.Hippmem(
  "localhost:50051",
  grpc.credentials.createInsecure()
);

// Write
client.write(
  { content: "The user prefers Rust for development and values clean error handling.", content_type: "Preference" },
  (err, resp) => {
    if (err) {
      console.error(err);
      return;
    }
    console.log(`write ok: memory_id=${resp.memory_id}`);

    // Retrieve (executed serially inside the write callback)
    client.retrieve(
      { query: "What language does the user prefer?", top_k: 5, mode: "Balanced" },
      (err, resp) => {
        if (err) {
          console.error(err);
          return;
        }
        resp.results.forEach((item) => {
          console.log(`[${item.score.toFixed(3)}] ${item.content.substring(0, 60)}`);
        });
      }
    );
  }
);
```

---

## gRPC vs In-process Rust API

| Dimension | gRPC | Direct Rust library call |
|------|------|----------------|
| **Performance** | Network serialization overhead | Zero copy, fastest |
| **Language support** | Any language supported by gRPC | Rust only |
| **Deployment** | Standalone process, horizontally scalable | Embedded in the same process |
| **Debugging** | grpcurl / grpcui | `println!` / `dbg!` |
| **Use cases** | Cross-language / microservices | Same-process Rust projects |

**Recommendation**: Use the `hippmem-engine` crate directly for Rust projects; call via gRPC for Python/Go/Node.js projects.

---

## References

- [API Reference](api-reference.md) — Rust API signatures and type definitions
- [Integration Guide](integration.md) — More deployment patterns
