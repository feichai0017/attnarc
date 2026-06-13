//! P/D proxy — the running service that joins the pieces: it puts the QuillCache
//! store on the request hot path by orchestrating a disaggregated prefill→decode
//! flow across two engines that share one store.
//!
//! For each `POST /v1/chat/completions`:
//!   1. send the prompt to the **prefill** engine (with `max_tokens=1`) — its
//!      QuillCache KV connector computes the prefix KV and offloads it to the
//!      shared store;
//!   2. send the original request to the **decode** engine — its connector finds
//!      that prefix in the store and loads it instead of re-prefilling, then
//!      generates; the decode response is returned to the caller.
//!
//! So the bytes really cross the store between two engines on the hot path
//! (gateway/proxy + store + connector as one service), rather than each piece
//! standing alone. True mid-request token-level P/D (vLLM's kv_producer/consumer
//! handshake) is the next step on top of this orchestration.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Clone)]
struct ProxyState {
    client: reqwest::Client,
    prefill_url: String,
    decode_url: String,
}

/// Orchestrate prefill (warm the store) → decode (reuse from the store).
async fn chat(
    State(st): State<Arc<ProxyState>>,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // 1) Warm: prefill the prompt with max_tokens=1 so the prefill engine just
    //    computes + offloads the prefix KV to the shared store.
    let mut warm = body.clone();
    if let Some(obj) = warm.as_object_mut() {
        obj.insert("max_tokens".into(), Value::from(1));
        obj.insert("stream".into(), Value::from(false));
    }
    st.client
        .post(format!("{}/v1/chat/completions", st.prefill_url))
        .json(&warm)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("prefill engine: {e}")))?;

    // 2) Generate: the decode engine finds the prefix KV in the store (its
    //    connector loads it instead of re-prefilling) and generates.
    let resp = st
        .client
        .post(format!("{}/v1/chat/completions", st.decode_url))
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("decode engine: {e}")))?;

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::OK);
    let text = resp
        .text()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    Ok((
        status,
        [
            (header::CONTENT_TYPE, "application/json"),
            (
                header::HeaderName::from_static("x-quillcache-pd"),
                "prefill→store→decode",
            ),
        ],
        text,
    ))
}

async fn state(State(st): State<Arc<ProxyState>>) -> Json<Value> {
    Json(serde_json::json!({
        "mode": "pd-proxy",
        "flow": "prefill → store → decode",
        "prefill": st.prefill_url,
        "decode": st.decode_url,
    }))
}

fn router(st: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/state", get(state))
        .with_state(st)
}

pub async fn run_pd_proxy(
    bind: String,
    prefill_url: String,
    decode_url: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let st = Arc::new(ProxyState {
        client: reqwest::Client::new(),
        prefill_url: prefill_url.trim_end_matches('/').to_string(),
        decode_url: decode_url.trim_end_matches('/').to_string(),
    });
    let socket: SocketAddr = bind.parse()?;
    let listener = TcpListener::bind(socket).await?;
    println!("QuillCache P/D proxy on http://{socket}  (prefill → store → decode)");
    println!("  prefill: {}   decode: {}", st.prefill_url, st.decode_url);
    axum::serve(listener, router(st)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State as AxState;
    use std::sync::Mutex;

    type Seen = Arc<Mutex<Vec<i64>>>;

    // A mock engine that records the max_tokens it received and echoes its name.
    async fn mock_engine(
        AxState((name, seen)): AxState<(String, Seen)>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let mt = body.get("max_tokens").and_then(|v| v.as_i64()).unwrap_or(-1);
        seen.lock().unwrap().push(mt);
        Json(serde_json::json!({"engine": name, "saw_max_tokens": mt}))
    }

    async fn spawn_mock(name: &str) -> (String, Seen) {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_engine))
            .with_state((name.to_string(), seen.clone()));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), seen)
    }

    #[tokio::test]
    async fn proxy_warms_prefill_then_returns_decode() {
        let (prefill_url, prefill_seen) = spawn_mock("prefill").await;
        let (decode_url, decode_seen) = spawn_mock("decode").await;

        let st = Arc::new(ProxyState {
            client: reqwest::Client::new(),
            prefill_url: prefill_url.clone(),
            decode_url: decode_url.clone(),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(st)).await.unwrap() });

        let http = reqwest::Client::new();
        let out: Value = http
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({"model":"m","messages":[],"max_tokens":16}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        // The caller gets the DECODE engine's response.
        assert_eq!(out["engine"], "decode");
        // Prefill was warmed with max_tokens=1; decode saw the original 16.
        assert_eq!(prefill_seen.lock().unwrap().as_slice(), &[1]);
        assert_eq!(decode_seen.lock().unwrap().as_slice(), &[16]);
    }
}
