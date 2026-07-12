//! HIPPMEM CLI — command-line tool.
//!
//! ```text
//! hippmem write -c "memory content" -t Decision
//! hippmem retrieve -q "query text" -k 5
//! hippmem explain --memory-id 12345
//! hippmem inspect store-stats
//! hippmem serve  # start the gRPC server
//! ```

use clap::{Parser, Subcommand};
use hippmem_core::config::EmbedderConfig;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{
    ConsolidationScope, DumpInput, Engine, EngineConfig, InspectQuery, ListInput, RetrieveContext,
    RetrieveInput, TraverseDirection, TraverseInput, WriteMemoryInput,
};
use std::path::PathBuf;

/// Entry point: auto-loads the project-root .env; does not error when the file is absent.
fn main() {
    let _ = dotenvy::dotenv();
    main_impl()
}

#[derive(Parser)]
#[command(
    name = "hippmem",
    version,
    about = "HIPPMEM Native Association Memory Engine"
)]
struct Cli {
    #[arg(
        short,
        long,
        env = "HIPPMEM_STORE_DIR",
        default_value = "./hippmem_data"
    )]
    store_dir: PathBuf,

    /// Embedding backend: deterministic (default) | openai-compatible (requires api-backends feature)
    #[arg(
        long,
        global = true,
        env = "HIPPMEM_EMBEDDING_PROVIDER",
        value_parser = ["deterministic", "openai-compatible"]
    )]
    embedding_provider: Option<String>,

    /// base URL of the OpenAI-compatible API (e.g. DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1)
    #[arg(long, global = true, env = "HIPPMEM_EMBEDDING_BASE_URL")]
    embedding_base_url: Option<String>,

    /// Model name of the OpenAI-compatible API (e.g. text-embedding-v4 / text-embedding-3-small)
    #[arg(long, global = true, env = "HIPPMEM_EMBEDDING_MODEL")]
    embedding_model: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum InspectCommand {
    /// Storage stats (total memory count, edge count, per-index sizes)
    #[command(name = "store-stats")]
    StoreStats,
    /// Background queue status
    #[command(name = "queue")]
    Queue,
    /// Single memory overview + in/out edges
    #[command(name = "memory")]
    Memory {
        /// Memory ID (u128 number)
        id: String,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// Write a memory
    Write {
        #[arg(short, long)]
        content: String,
        #[arg(short = 't', long, default_value = "UserStatement")]
        content_type: String,
    },
    /// Retrieve memories
    Retrieve {
        #[arg(short, long)]
        query: String,
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
    },
    /// Explain a memory
    Explain {
        #[arg(short, long)]
        memory_id: u128,
    },
    /// List memories with pagination
    List {
        #[arg(long, default_value = "20")]
        limit: usize,
        #[arg(long)]
        cursor: Option<u128>,
        #[arg(short = 't', long)]
        content_type: Option<String>,
    },
    /// Run consolidation (optional --scope incremental|reindex|full)
    Consolidate {
        #[arg(long, default_value = "incremental")]
        scope: String,
    },
    /// Diagnostics
    Inspect {
        #[command(subcommand)]
        command: Option<InspectCommand>,
    },
    /// Start the gRPC server (requires --features grpc)
    Serve,
    /// Clear all data (deletes the redb database and the Tantivy fulltext index; irreversible)
    Clear,
    /// Full export of memories as JSONL
    Dump {
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Graph traversal: start from a memory and explore along association edges
    Traverse {
        #[arg(long)]
        from: u128,
        #[arg(long, default_value = "2")]
        depth: u8,
        #[arg(long, default_value = "outgoing")]
        direction: String,
    },
}

fn main_impl() {
    let cli = Cli::parse();

    // ── Clear subcommand: do not open the Engine; delete files directly (idempotent) ──
    if matches!(cli.command, Commands::Clear) {
        let store_path = &cli.store_dir;
        let fulltext_path = store_path
            .parent()
            .map(|p| p.join("fulltext"))
            .unwrap_or_else(|| PathBuf::from("hippmem_data").join("fulltext"));

        let mut cleared = false;

        // Delete the redb database file (idempotent)
        if store_path.exists() {
            if store_path.is_file() {
                std::fs::remove_file(store_path).unwrap_or_else(|e| {
                    eprintln!("⚠ cannot delete {}: {}", store_path.display(), e)
                });
            } else {
                std::fs::remove_dir_all(store_path).unwrap_or_else(|e| {
                    eprintln!("⚠ cannot delete {}: {}", store_path.display(), e)
                });
            }
            cleared = true;
        }

        // Delete the Tantivy fulltext index directory (idempotent)
        if fulltext_path.exists() {
            std::fs::remove_dir_all(&fulltext_path).unwrap_or_else(|e| {
                eprintln!("⚠ cannot delete {}: {}", fulltext_path.display(), e)
            });
            cleared = true;
        }

        if cleared {
            println!(
                "✓ cleared: {} and {}",
                store_path.display(),
                fulltext_path.display()
            );
        } else {
            println!(
                "✓ data directory does not exist, nothing to clear ({})",
                store_path.display()
            );
        }
        return;
    }

    // Build EmbedderConfig from CLI args (CLI > env vars > defaults)
    let embedder = match cli.embedding_provider.as_deref() {
        Some("openai-compatible") => EmbedderConfig::OpenAiCompatible {
            base_url: cli
                .embedding_base_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".into()),
            model: cli
                .embedding_model
                .clone()
                .unwrap_or_else(|| "text-embedding-v4".into()),
            api_key: None,    // read by build_embedder from the OPENAI_API_KEY env var
            dimensions: 1024, // DashScope text-embedding-v4 default dimensions
        },
        _ => EmbedderConfig::default(),
    };

    let config = EngineConfig {
        store_dir: cli.store_dir.clone(),
        embedder,
        ..Default::default()
    };
    let engine = Engine::open(config).expect("Failed to open engine");

    match cli.command {
        Commands::Write {
            content,
            content_type,
        } => {
            let ct = parse_content_type(&content_type);
            let out = engine
                .write(WriteMemoryInput {
                    content,
                    content_type: Some(ct),
                    context: WriteContext {
                        conversation_id: None,
                        session_id: None,
                        project_id: None,
                        task_id: None,
                        user_id: None,
                        local_time: hippmem_core::time::Timestamp(0),
                        preceding_memory_ids: vec![],
                        source_refs: vec![],
                    },
                    importance_hint: None,
                    source_refs: vec![],
                })
                .expect("write failed");
            println!(
                "✓ memory_id: {} stage: {:?} links: {}",
                out.memory_id.0,
                out.stage_reached,
                out.created_links.len()
            );
        }
        Commands::Retrieve { query, top_k } => {
            let out = engine
                .retrieve(RetrieveInput {
                    query,
                    context: RetrieveContext::default(),
                    top_k,
                    max_hops: Some(2),
                    retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
                })
                .expect("retrieve failed");
            for (i, r) in out.results.iter().enumerate() {
                println!(
                    "{}. [{:.3}] {} (dims: {:?})",
                    i + 1,
                    r.final_score,
                    r.memory.content.raw.chars().take(80).collect::<String>(),
                    r.matched_dimensions
                );
            }
            if out.results.is_empty() {
                println!("(no results)");
            }
        }
        Commands::Explain { memory_id } => {
            let e = engine
                .explain(hippmem_core::ids::MemoryId(memory_id), None)
                .expect("explain failed");
            println!(
                "memory: {} importance: {:.3} links: {} corrections: {}",
                e.content_summary,
                e.current_importance,
                e.linked.len(),
                e.corrections.len()
            );
        }
        Commands::List {
            limit,
            cursor,
            content_type,
        } => {
            let ct = content_type.as_deref().map(parse_content_type);
            let out = engine
                .list(ListInput {
                    limit,
                    cursor,
                    content_type: ct,
                })
                .expect("list failed");

            if out.items.is_empty() {
                println!("(no memories)");
            } else {
                // Header
                let header = format!(
                    "{:<14} {:<18} {:>10} {:>5}  {}",
                    "ID", "Type", "Importance", "Edges", "Preview"
                );
                println!("{}", header);
                println!(
                    "{:-<14} {:-<18} {:-<10} {:-<5}  {:-<60}",
                    "", "", "", "", ""
                );

                for item in &out.items {
                    // Take the last 12 chars of the ID (the random part of ULID is at the end; the prefix is similar when timestamps are close)
                    let id_str = format!("{}", item.id.0);
                    let id_short = if id_str.len() > 12 {
                        &id_str[id_str.len() - 12..]
                    } else {
                        &id_str
                    };
                    let type_str = format!("{:?}", item.content_type);
                    let preview: String = item.content_preview.chars().take(60).collect();
                    println!(
                        "{:<14} {:<18} {:>10.3} {:>5}  {}",
                        id_short, type_str, item.importance, item.edge_count, preview
                    );
                }

                println!();
                print!("── {} total", out.total);
                if let Some(c) = out.next_cursor {
                    println!(", next cursor: {}", c);
                } else {
                    println!(" (last page)");
                }
            }
        }
        Commands::Consolidate { scope } => {
            let scope = match scope.as_str() {
                "reindex" => ConsolidationScope::Reindex,
                "full" => ConsolidationScope::Full,
                _ => ConsolidationScope::Incremental,
            };
            let r = engine.consolidate(scope).expect("consolidate failed");
            println!(
                "processed: {} decayed: {} reindexed: {} elapsed: {}ms",
                r.memories_processed, r.edges_decayed, r.reindexed, r.elapsed_ms
            );
        }
        Commands::Inspect { command } => match command {
            Some(InspectCommand::StoreStats) | None => {
                let r = engine
                    .inspect(InspectQuery::StoreStats)
                    .expect("inspect failed");
                match r {
                    hippmem_engine::InspectReport::StoreStats(s) => {
                        println!(
                            "memories: {} edges: {} backlog: {}",
                            s.memory_count, s.edge_count, s.queue_backlog
                        );
                    }
                    _ => println!("unexpected variant"),
                }
            }
            Some(InspectCommand::Queue) => {
                let r = engine
                    .inspect(InspectQuery::QueueStatus)
                    .expect("inspect failed");
                match r {
                    hippmem_engine::InspectReport::QueueStatus(q) => {
                        println!(
                            "pending: enrich={}, consolidate={}, in_flight={}",
                            q.pending_enrich, q.pending_consolidate, q.in_flight
                        );
                    }
                    _ => println!("unexpected variant"),
                }
            }
            Some(InspectCommand::Memory { id }) => {
                let memory_id = match id.parse::<u128>() {
                    Ok(v) => hippmem_core::ids::MemoryId(v),
                    Err(_) => {
                        eprintln!("error: invalid memory ID '{}', expected a u128 number", id);
                        return;
                    }
                };
                let r = engine
                    .inspect(InspectQuery::Memory(memory_id))
                    .expect("inspect failed");
                match r {
                    hippmem_engine::InspectReport::Memory(m) => {
                        println!("━━ Memory detail ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                        println!("ID:           {}", m.unit.id.0);
                        println!(
                            "Content:      {}",
                            m.unit.content.raw.chars().take(200).collect::<String>()
                        );
                        println!("Type:         {:?}", m.unit.content.content_type);
                        println!("Stage:        {:?}", m.stage);
                        println!("Lifecycle:    {:?}", m.lifecycle);
                        println!(
                            "Importance:   {:.3}",
                            m.unit.understanding.importance.value()
                        );
                        println!("Retrievals:   {}", m.unit.activation.retrieval_count);

                        // Out-edges
                        println!();
                        if m.out_edges.is_empty() {
                            println!("━━ Out-edges (0) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            println!("  (none)");
                        } else {
                            println!(
                                "━━ Out-edges ({}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                m.out_edges.len()
                            );
                            for e in &m.out_edges {
                                let preview: String = e.evidence.chars().take(60).collect();
                                println!(
                                    "  → {:<14} [{:<20}] strength={:.3}  {}",
                                    e.to.0,
                                    format!("{:?}", e.link_type),
                                    e.strength,
                                    preview
                                );
                            }
                        }

                        // In-edges
                        if m.in_edges.is_empty() {
                            println!("━━ In-edges (0) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            println!("  (none)");
                        } else {
                            println!(
                                "━━ In-edges ({}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                m.in_edges.len()
                            );
                            for e in &m.in_edges {
                                let preview: String = e.evidence.chars().take(60).collect();
                                println!(
                                    "  ← {:<14} [{:<20}] strength={:.3}  {}",
                                    e.from.0,
                                    format!("{:?}", e.link_type),
                                    e.strength,
                                    preview
                                );
                            }
                        }
                    }
                    _ => println!("unexpected variant"),
                }
            }
        },
        Commands::Dump { output } => {
            let out = engine
                .dump(DumpInput {
                    output_path: output.clone(),
                })
                .expect("dump failed");
            if let Some(path) = out.written_to {
                println!("✓ exported {} memories to {}", out.count, path.display());
            } else if let Some(json) = out.json {
                print!("{}", json);
            }
        }
        Commands::Traverse {
            from,
            depth,
            direction,
        } => {
            let dir = match direction.as_str() {
                "incoming" => TraverseDirection::Incoming,
                "both" => TraverseDirection::Both,
                _ => TraverseDirection::Outgoing,
            };
            let out = engine
                .traverse(TraverseInput {
                    start_id: hippmem_core::ids::MemoryId(from),
                    max_depth: depth,
                    direction: dir,
                    link_types: None, // not exposed in CLI
                })
                .expect("traverse failed");

            println!(
                "━━ Graph traversal: from {} (BFS, max_depth={}, {:?}) ━━",
                from, depth, dir
            );
            println!();

            if out.nodes.is_empty() {
                println!("  (no neighbors)");
            } else {
                for node in &out.nodes {
                    let preview: String = node.content_preview.chars().take(60).collect();
                    println!(
                        "  [depth={}] {:<14} importance={:.3}  {}",
                        node.depth, node.id.0, node.importance, preview
                    );
                }
            }

            println!();
            println!("── {} nodes, {} edges", out.nodes.len(), out.edges.len());
        }
        Commands::Serve => {
            println!("gRPC server not yet implemented (use --features grpc)");
        }
        Commands::Clear => unreachable!("Clear handled before engine open"),
    }

    engine.close().expect("close failed");
}

fn parse_content_type(s: &str) -> ContentType {
    match s.to_lowercase().as_str() {
        "userstatement" => ContentType::UserStatement,
        "decision" => ContentType::Decision,
        "preference" => ContentType::Preference,
        "event" => ContentType::Event,
        "taskstate" => ContentType::TaskState,
        "projectknowledge" => ContentType::ProjectKnowledge,
        "reflection" => ContentType::Reflection,
        "correction" => ContentType::Correction,
        _ => ContentType::UserStatement,
    }
}
