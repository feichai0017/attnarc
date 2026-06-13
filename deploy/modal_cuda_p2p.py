"""Verify the zero-copy GPU↔GPU peer data path (no NIC, no RDMA) on 2 GPUs.

This is the no-RDMA-hardware answer to Mooncake's zero-copy transfer: within a
node, `cuMemcpyPeer` moves bytes HBM→HBM directly over NVLink / PCIe-P2P, with no
host staging — the same hop GPUDirect-RDMA removes across nodes. It exercises the
real peer-copy added to `DeviceSegment` (transport/device_segment.rs):

  - peer_copy_moves_bytes_gpu0_to_gpu1   : correctness — bytes registered on GPU 0
    land in GPU 1's segment and read back identically (HBM→HBM, no host bounce)
  - peer_copy_bandwidth_vs_host_staged   : the win — peer copy vs D2H+H2D staging,
    prints `QC-P2P ... peer=..GB/s staged=..GB/s speedup=..x`

    modal run deploy/modal_cuda_p2p.py

cudarc's dynamic-loading dlopen's `libcuda.so`; a GPU container ships
`libcuda.so.1`, so we symlink the unversioned name onto LD_LIBRARY_PATH first.
2×L4 is PCIe-P2P (peer copy is 1 hop vs staging's 2, so still a win); an
NVLink box (A100:2 / H100:2) shows a much larger speedup.
"""

import modal

GPU = "L4:2"  # PCIe-P2P; swap to "A100:2" for the NVLink number

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .add_local_file("Cargo.toml", "/build/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", "/build/Cargo.lock", copy=True)
    .add_local_dir("crates", "/build/crates", copy=True, ignore=["**/target/**"])
    .add_local_dir("src", "/build/src", copy=True, ignore=["**/target/**"])
    .run_commands(
        "cd /build && $HOME/.cargo/bin/cargo test -p quillcache-transfer-engine --features cuda --no-run",
    )
)
app = modal.App("quillcache-cuda-p2p")


@app.function(gpu=GPU, image=image, timeout=60 * 20)
def verify():
    import glob
    import os
    import subprocess

    # Expose unversioned libcuda.so where cudarc's dlopen looks.
    linkdir = "/usr/local/lib/quillcache-cuda"
    os.makedirs(linkdir, exist_ok=True)
    cands = []
    for p in [
        "/usr/lib/x86_64-linux-gnu/libcuda.so*",
        "/usr/lib64/libcuda.so*",
        "/usr/local/cuda/lib64/libcuda.so*",
    ]:
        cands += glob.glob(p)
    cands = sorted(c for c in cands if os.path.basename(c) != "libcuda.so")
    if cands:
        link = os.path.join(linkdir, "libcuda.so")
        if not os.path.exists(link):
            os.symlink(cands[-1], link)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = linkdir + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    )

    suites = [
        ("peer copy correctness (GPU0→GPU1, HBM→HBM)", "peer_copy_moves_bytes_gpu0_to_gpu1"),
        ("peer copy bandwidth vs host-staged", "peer_copy_bandwidth_vs_host_staged"),
    ]
    results = []
    for label, test in suites:
        run = subprocess.run(
            ["cargo", "test", "-p", "quillcache-transfer-engine", "--features", "cuda", test,
             "--", "--ignored", "--nocapture", "--test-threads=1"],
            cwd="/build", env=env, capture_output=True, text=True,
        )
        results.append({
            "label": label,
            "stdout": run.stdout[-2000:], "stderr": run.stderr[-2000:],
            "returncode": run.returncode,
        })
    return {"gpu": smi.stdout.strip(), "suites": results}


@app.local_entrypoint()
def main():
    res = verify.remote()
    print("\n" + "=" * 78)
    print("QuillCache zero-copy GPU↔GPU peer data path (cuMemcpyPeer) — real GPU verification")
    print("=" * 78)
    print("GPU:", res["gpu"])
    all_ok = True
    p2p_line = None
    for s in res["suites"]:
        ok = s["returncode"] == 0
        all_ok = all_ok and ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {s['label']}")
        for line in (s["stdout"] + "\n" + s["stderr"]).splitlines():
            if "test result" in line or " ... " in line or line.startswith("running"):
                print("   ", line.strip())
            if "QC-P2P" in line:
                p2p_line = line.strip()
        if not ok:
            print("   --- stderr ---")
            print(s["stderr"])
    if p2p_line:
        print("\nbandwidth:", p2p_line)
    print(
        f"\nVERDICT: {'PASS — zero-copy HBM→HBM peer transfer works on real GPUs; the host-staged hop is removed (the no-NIC equivalent of Mooncake GPUDirect-RDMA)' if all_ok else 'see failures above'}"
    )
