//! Disaggregated prefill/decode demo — Mooncake's P/D data path, end to end and
//! real (no GPU). The control plane decides the plan (which worker prefills,
//! which decodes); then the freshly-prefilled KV physically moves from the
//! prefill node to the decode node **over the Transfer Engine**, identity-guarded.
//!
//! This wires three real pieces together:
//!   1. `ControlPlane::plan` → a `Disaggregated` `RequestPlan` (`prefill_worker`,
//!      `decode_worker`, `RunPrefill` blocks) and its `KvHandoff`.
//!   2. the prefill worker publishes the computed KV to the store
//!      (`put_start` → WRITE over the transfer engine → `put_end`).
//!   3. the decode worker resolves the replica (identity-guarded `get_replica_list`)
//!      and READs it over the transfer engine before continuing generation.
//!
//! The bytes really cross loopback TCP between two transfer engines; swap
//! `TcpTransport` for RDMA / the local engines for remote nodes and nothing above
//! changes. On a GPU the KV would be the prefill engine's paged-KV (the connector
//! seam); here it is a stand-in buffer so the *handoff* is exercised without a GPU.

use quillcache_core::{
    ControlPlane, EngineEndpoint, EngineKind, EngineRole, IdentityScope, KvBlockKey, RequestShape,
    ServingMode, SloTarget,
};
use quillcache_store::{ErrorCode, RealClient, ReplicateConfig};
use quillcache_transfer_engine::{InMemoryMetadata, MetadataBackend, TransferEngine};
use std::sync::Arc;

fn endpoint(id: &str, role: EngineRole) -> EngineEndpoint {
    EngineEndpoint {
        id: id.to_string(),
        kind: EngineKind::Vllm,
        role,
        base_url: "http://127.0.0.1:0".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        tokenizer_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        tenant_id: "tenant-a".to_string(),
        locality_domain: "local".to_string(),
    }
}

fn identity_of(req: &RequestShape) -> IdentityScope {
    IdentityScope {
        model_id: req.model_id.clone(),
        tokenizer_id: req.tokenizer_id.clone(),
        adapter_id: req.adapter_id.clone(),
        tenant_id: req.tenant_id.clone(),
    }
}

pub async fn run_pd_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuillCache P/D — disaggregated prefill→decode KV handoff (Mooncake data path)\n");

    // 1) Control plane: one prefill-only worker + one decode-only worker. A cold
    //    prefix block (not resident anywhere) forces a prefill the decode worker
    //    cannot do itself → the planner disaggregates.
    let control = ControlPlane::new(vec![
        endpoint("prefill-a", EngineRole::Prefill),
        endpoint("decode-a", EngineRole::Decode),
    ]);
    let request = RequestShape {
        id: "req-pd-0".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        tokenizer_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        adapter_id: None,
        tenant_id: "tenant-a".to_string(),
        session_id: None,
        blocks: vec![KvBlockKey::new(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "tenant-a",
            "root",
            "blk-cold",
            0,
            64,
        )],
        estimated_decode_tokens: 16,
        slo: SloTarget::default(),
    };

    let plan = control.plan(&request)?;
    let handoff = plan
        .kv_handoff()
        .ok_or("expected a disaggregated plan (got aggregated)")?;
    assert_eq!(plan.mode, ServingMode::Disaggregated);

    println!("  control-plane decision:");
    println!(
        "    request {} (tenant-a) → mode = {:?}",
        request.id, plan.mode
    );
    println!(
        "    prefill worker: {}   decode worker: {}",
        handoff.prefill_worker_id, handoff.decode_worker_id
    );
    println!(
        "    handoff: {} freshly-prefilled block(s) prefill computes → decode must receive\n",
        handoff.blocks.len()
    );

    // 2) The transfer-engine reality: the prefill worker's node serves a segment
    //    (where its computed KV lives); the decode worker's node reads from it.
    let metadata: Arc<dyn MetadataBackend> = Arc::new(InMemoryMetadata::new());
    let _prefill_node = TransferEngine::init("prefill-a", metadata.clone(), "127.0.0.1:0").await?;
    let decode_engine = TransferEngine::init("decode-a", metadata.clone(), "127.0.0.1:0").await?;

    // A client driving the data path; it reads/writes the prefill node's segment
    // over the transfer engine (decode-a's engine is the one issuing the READ).
    let mut client = RealClient::new("random", decode_engine);
    client.mount("prefill-a", 1 << 20);

    let id = identity_of(&request);
    let key = format!("kv-handoff:{}", handoff.request_id);

    // 2a) prefill-a computes the block's KV (stand-in bytes) and publishes it to
    //     the pool: put_start (master allocates on prefill-a) → WRITE → put_end.
    let kv = vec![0xA5u8; 4096];
    client
        .put(&key, id.clone(), &kv, &ReplicateConfig::replicas(1))
        .await?;
    println!("  data path (real bytes over the transfer engine):");
    println!(
        "    prefill-a computed {} B of KV and published it to the pool (put_start→WRITE→put_end)",
        kv.len()
    );

    // 2b) decode-a resolves the replica (identity-guarded) and READs it back over
    //     the transfer engine — the KV crosses from the prefill node to decode.
    let received = client.get(&key, &id).await?;
    let ok = received.len() == kv.len() && received[..] == kv[..];
    println!(
        "    decode-a resolved the replica (identity-guarded) and READ {} B over the engine → {}",
        received.len(),
        if ok {
            "matches prefill output"
        } else {
            "MISMATCH (bug!)"
        }
    );

    // 3) The identity guard: a different tenant cannot pull tenant-a's KV.
    let refused = matches!(
        client
            .get(
                &key,
                &IdentityScope {
                    tenant_id: "tenant-b".into(),
                    ..id.clone()
                },
            )
            .await,
        Err(ErrorCode::UnsafeReuse(_))
    );
    println!(
        "    identity guard: tenant-b's decode of tenant-a's KV was {}",
        if refused {
            "REFUSED (no cross-tenant leak)"
        } else {
            "SERVED (bug!)"
        }
    );
    println!(
        "\n  master: {} object(s) across {} segment(s)",
        client.master().object_count(),
        client.master().segment_count()
    );

    if ok && refused {
        println!("\n  P/D handoff complete: control-plane plan → real prefill→decode KV transfer.");
        Ok(())
    } else {
        Err("P/D handoff verification failed".into())
    }
}
