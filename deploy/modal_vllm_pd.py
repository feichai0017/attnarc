"""GPU-real disaggregated prefill/decode — Mooncake's P/D, end to end on GPUs.

Two SEPARATE vLLM instances in one 2-GPU container, sharing one QuillCache store:
  - prefill  : vLLM on GPU 0 (port 8000), QuillCacheV1Connector
  - decode   : vLLM on GPU 1 (port 8001), QuillCacheV1Connector
  - store    : one quillcache store-master + transfer-node on localhost, shared

The flow is Mooncake's P/D: the prompt goes to the PREFILL instance, which
computes the prefix KV and publishes it to the store; then the SAME prompt goes
to the DECODE instance, which finds prefill's KV in the store and LOADS it over
the transfer engine instead of prefilling — KV computed on GPU 0 is reused on
GPU 1, across instances. Prefix caching is off on both so the only way decode
can skip prefill is via the connector.

    modal run deploy/modal_vllm_pd.py

(One 2-GPU box avoids Modal cross-container networking; swap the localhost store
for a shared remote store + put the two vLLMs on two nodes and nothing changes.)
"""

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .pip_install("vllm", "huggingface_hub")
    .add_local_file("Cargo.toml", "/build/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", "/build/Cargo.lock", copy=True)
    .add_local_dir("crates", "/build/crates", copy=True, ignore=["**/target/**"])
    .add_local_dir("src", "/build/src", copy=True, ignore=["**/target/**"])
    .run_commands(
        "cd /build && $HOME/.cargo/bin/cargo build --release --bin quillcache",
        "cp /build/target/release/quillcache /usr/local/bin/quillcache",
    )
    .add_local_file("bridge/quillcache_v1_connector.py", "/root/quillcache_v1_connector.py", copy=True)
    .add_local_file("bridge/quillcache_store_client.py", "/root/quillcache_store_client.py", copy=True)
)
app = modal.App("quillcache-vllm-pd")


@app.function(gpu="L4:2", image=image, timeout=60 * 60)
def run_pd():
    import json
    import os
    import subprocess
    import time
    import urllib.request

    procs = {}

    def tail(path, n=60):
        try:
            return "\n".join(open(path, errors="replace").read().splitlines()[-n:])
        except OSError:
            return f"<no {path}>"

    def spawn(name, args, logpath, extra_env=None):
        env = dict(os.environ)
        env["PYTHONPATH"] = "/root"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        if extra_env:
            env.update(extra_env)
        f = open(logpath, "w")
        procs[name] = (subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env), f)

    def wait_ready(url, timeout, name, proc=None):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"{name} exited early (code {proc.returncode})")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status < 500:
                        return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError(f"{name} not ready at {url} within {timeout}s")

    def kv_cfg():
        return json.dumps({
            "kv_connector": "QuillCacheV1Connector",
            "kv_connector_module_path": "quillcache_v1_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "master_url": "http://127.0.0.1:7777",
                "segment_endpoints": {"seg-0": "127.0.0.1:8100"},
                "tenant_id": "default",
                "replica_num": 1,
            },
        })

    def serve(port, gpu_ordinal, logpath):
        spawn(
            f"vllm-{port}",
            [
                "vllm", "serve", MODEL,
                "--port", str(port),
                "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.55",
                "--no-enable-prefix-caching",
                "--disable-hybrid-kv-cache-manager",
                "--kv-transfer-config", kv_cfg(),
            ],
            logpath,
            extra_env={"CUDA_VISIBLE_DEVICES": str(gpu_ordinal)},
        )

    def chat(port):
        shared_prefix = (
            "You are a meticulous assistant. Follow these standing rules for every "
            "answer. " + " ".join(
                f"Rule {i}: always be precise, cite assumptions, and prefer concrete "
                "examples over abstractions." for i in range(24)
            )
        )
        body = json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": shared_prefix},
                {"role": "user", "content": "In one sentence, what is a KV cache?"},
            ],
            "max_tokens": 16,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
        return (time.time() - t0) * 1000, out["choices"][0]["message"]["content"]

    try:
        # 1) shared store: one master + one transfer-node segment on localhost.
        spawn("transfer-node", ["quillcache", "transfer-node", "--addr", "127.0.0.1:8100", "--segment", "seg-0"], "/tmp/node.log")
        spawn("store-master", ["quillcache", "store-master", "--addr", "127.0.0.1:7777"], "/tmp/master.log")
        wait_ready("http://127.0.0.1:7777/v1/state", 30, "store-master", procs["store-master"][0])

        # 2) two vLLM instances: prefill on GPU 0, decode on GPU 1, same store.
        serve(8000, 0, "/tmp/vllm_prefill.log")
        serve(8001, 1, "/tmp/vllm_decode.log")
        wait_ready("http://127.0.0.1:8000/health", 900, "vllm-prefill", procs["vllm-8000"][0])
        wait_ready("http://127.0.0.1:8001/health", 900, "vllm-decode", procs["vllm-8001"][0])

        # 3) prefill the prompt on GPU 0 → its KV is published to the store.
        t_prefill, out_p = chat(8000)
        time.sleep(2.0)  # let the manifest commit settle
        # 4) same prompt to the DECODE instance on GPU 1 → it loads prefill's KV
        #    from the store instead of prefilling.
        t_decode, out_d = chat(8001)
        time.sleep(1.0)

        prefill_log = open("/tmp/vllm_prefill.log", errors="replace").read()
        decode_log = open("/tmp/vllm_decode.log", errors="replace").read()

        def grep(text, needle):
            return [l for l in text.splitlines() if needle in l]

        prefill_committed = grep(prefill_log, "QC committed")
        decode_hit = [l for l in grep(decode_log, "QC match-check") if "manifest=True" in l]
        decode_loaded = grep(decode_log, "QC loading")
        state = json.loads(urllib.request.urlopen("http://127.0.0.1:7777/v1/state").read())

        result = {
            "ok": True,
            "prefill_ms": round(t_prefill),
            "decode_ms": round(t_decode),
            "outputs_match": out_p == out_d,
            "prefill_committed": prefill_committed[:3],
            "decode_manifest_hit": decode_hit[:3],
            "decode_loaded": decode_loaded[:3],
            "store_state": state,
            "decode_log_tail": "\n".join(decode_log.splitlines()[-30:]),
        }
    except Exception as e:
        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "prefill_log_tail": tail("/tmp/vllm_prefill.log"),
            "decode_log_tail": tail("/tmp/vllm_decode.log"),
            "master_log_tail": tail("/tmp/master.log", 15),
        }
    finally:
        for _, (p, f) in procs.items():
            try:
                p.terminate()
            except Exception:
                pass
            f.close()
    return result


@app.local_entrypoint()
def main():
    import json

    res = run_pd.remote()
    print("\n" + "=" * 80)
    print("QuillCache disaggregated P/D — prefill (GPU 0) → store → decode (GPU 1)")
    print("=" * 80)
    if not res.get("ok"):
        print(f"\nFAILED: {res.get('error')}")
        print("\n--- prefill log tail ---\n" + res.get("prefill_log_tail", ""))
        print("\n--- decode log tail ---\n" + res.get("decode_log_tail", ""))
        print("\n--- master log tail ---\n" + res.get("master_log_tail", ""))
        return
    print(f"\nlatency: prefill(GPU0)={res['prefill_ms']}ms  decode(GPU1)={res['decode_ms']}ms")
    print(f"outputs identical: {res['outputs_match']}")
    print("\n[prefill GPU0 committed KV to the store]")
    for l in res["prefill_committed"]:
        print("   ", l.strip()[:150])
    print("\n[decode GPU1 found prefill's KV in the store]")
    for l in res["decode_manifest_hit"]:
        print("   ", l.strip()[:150])
    print("\n[decode GPU1 loaded it over the transfer engine]")
    for l in res["decode_loaded"]:
        print("   ", l.strip()[:150])
    print("\n--- store master /v1/state ---")
    print(json.dumps(res["store_state"], indent=2)[:600])
    cross = bool(res["prefill_committed"]) and bool(res["decode_manifest_hit"]) and bool(res["decode_loaded"])
    print(
        f"\nVERDICT: {'REAL cross-instance P/D — GPU0 prefill KV reused by GPU1 decode via the store' if cross else 'partial — see decode_log_tail'}"
    )
    if not cross:
        print("\n--- decode log tail ---\n" + res["decode_log_tail"])
