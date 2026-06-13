//! CUDA device segment — GPU HBM registered as a Transfer Engine segment.
//!
//! Mooncake registers memory *segments* (host OR device) and moves bytes
//! one-sidedly by `(offset, length)`. This is the **device** form: the segment's
//! bytes live in GPU HBM (`cudaMalloc`), and READ / WRITE stage through
//! `cudaMemcpy` (D2H / H2D). It speaks the exact same wire as the host TCP
//! segment ([`crate::transport::tcp`]), so an unmodified
//! [`crate::transport::tcp::TcpTransport`] peer can READ & WRITE a GPU-resident
//! segment with no changes above the [`crate::transport::Transport`] trait.
//!
//! The host hop (D2H to serve a read, H2D to apply a write) is exactly what
//! GPUDirect-RDMA / NVLink (the reserved `nvlink` feature) removes — NIC / GPU ↔
//! HBM with no staging. This `cuda` path is the real, single-GPU-verifiable core;
//! `nvlink` is the zero-copy network optimization layered on top.
//!
//! Built only with `--features cuda` (cudarc with `dynamic-loading`, so it
//! compiles without CUDA present and runs on a GPU box).

use std::sync::{Arc, Mutex};

use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// Same wire as transport/tcp.rs.
const OP_READ: u8 = 0;
const OP_WRITE: u8 = 1;
const ST_OK: u8 = 0;
const ST_ERR: u8 = 2;

/// A GPU HBM byte arena registered as a transfer-engine segment. Capacity is
/// fixed (HBM is precious); a logical length grows on WRITE up to capacity,
/// mirroring the host segment's grow-on-write.
pub struct DeviceSegment {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    arena: Mutex<Arena>,
    capacity: usize,
}

struct Arena {
    hbm: CudaSlice<u8>,
    len: usize,
}

impl std::fmt::Debug for DeviceSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSegment")
            .field("ordinal", &self.ctx.ordinal())
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish()
    }
}

impl DeviceSegment {
    /// Bind GPU `ordinal` and reserve `capacity` bytes of HBM for the segment.
    pub fn new(ordinal: usize, capacity: usize) -> Result<Arc<Self>, String> {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new: {e:?}"))?;
        let stream = ctx.default_stream();
        let hbm = stream
            .alloc_zeros::<u8>(capacity)
            .map_err(|e| format!("alloc HBM arena: {e:?}"))?;
        Ok(Arc::new(Self {
            ctx,
            stream,
            arena: Mutex::new(Arena { hbm, len: 0 }),
            capacity,
        }))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.arena.lock().unwrap().len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Which CUDA device the segment's HBM lives on.
    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    /// Register a buffer: H2D-copy it into the arena at the current end; returns
    /// its offset — the handle a peer targets.
    pub fn register(&self, data: &[u8]) -> Result<u64, String> {
        let mut a = self.arena.lock().unwrap();
        let offset = a.len;
        self.write_locked(&mut a, offset, data)?;
        Ok(offset as u64)
    }

    /// READ: D2H-copy `len` bytes from HBM at `offset`. Errors if out of bounds.
    pub fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>, String> {
        let a = self.arena.lock().unwrap();
        let end = offset.checked_add(len).ok_or("offset+len overflow")?;
        if end > a.len {
            return Err("read out of segment bounds".into());
        }
        let mut host = vec![0u8; len];
        if len > 0 {
            let view = a.hbm.slice(offset..end);
            self.stream
                .memcpy_dtoh(&view, &mut host)
                .map_err(|e| format!("d2h: {e:?}"))?;
            self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        }
        Ok(host)
    }

    /// WRITE: H2D-copy `data` into HBM at `offset`, growing the logical length up
    /// to capacity. Errors if it would exceed capacity.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<(), String> {
        let mut a = self.arena.lock().unwrap();
        self.write_locked(&mut a, offset, data)
    }

    fn write_locked(&self, a: &mut Arena, offset: usize, data: &[u8]) -> Result<(), String> {
        let end = offset
            .checked_add(data.len())
            .ok_or("offset+len overflow")?;
        if end > self.capacity {
            return Err(format!(
                "write exceeds HBM segment capacity ({end} > {})",
                self.capacity
            ));
        }
        if !data.is_empty() {
            let mut view = a.hbm.slice_mut(offset..end);
            self.stream
                .memcpy_htod(data, &mut view)
                .map_err(|e| format!("h2d: {e:?}"))?;
            self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        }
        if end > a.len {
            a.len = end;
        }
        Ok(())
    }

    // ==============================
    // Zero-copy GPU↔GPU peer transfer (intra-node NVLink / PCIe P2P)
    //
    // The READ/WRITE path above stages every byte through host RAM (D2H to serve,
    // H2D to apply) — the exact hop GPUDirect-RDMA removes across nodes. *Within*
    // a node, CUDA peer copy (`cuMemcpyPeer`) removes the same hop: HBM→HBM
    // directly over NVLink/PCIe, no host bounce. This is the no-NIC equivalent of
    // Mooncake's zero-copy data path, and is the real, multi-GPU-verifiable core
    // of the `nvlink` transport.
    // ==============================

    /// Can this segment's GPU directly reach `peer`'s HBM (NVLink / PCIe P2P)?
    /// When false, [`Self::copy_to_peer`] still works — the driver stages through
    /// host internally — it just isn't the zero-copy fast path.
    pub fn can_access_peer(&self, peer: &DeviceSegment) -> Result<bool, String> {
        let mut can: std::ffi::c_int = 0;
        let res = unsafe {
            sys::cuDeviceCanAccessPeer(&mut can, self.ctx.cu_device(), peer.ctx.cu_device())
        };
        if res != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuDeviceCanAccessPeer: {res:?}"));
        }
        Ok(can != 0)
    }

    /// Grant this segment's context direct access to `peer`'s HBM, so a later
    /// [`Self::copy_to_peer`] is true HBM→HBM P2P instead of host-staged.
    /// `cuCtxEnablePeerAccess` acts on the *current* context, so we bind ours
    /// first. Idempotent: an already-enabled link is treated as success.
    pub fn enable_peer_access(&self, peer: &DeviceSegment) -> Result<(), String> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| format!("bind ctx: {e:?}"))?;
        let res = unsafe { sys::cuCtxEnablePeerAccess(peer.ctx.cu_ctx(), 0) };
        match res {
            sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Ok(())
            }
            other => Err(format!("cuCtxEnablePeerAccess: {other:?}")),
        }
    }

    /// Zero-copy device-to-device copy: `len` bytes from this segment at
    /// `src_offset` into `dst` at `dst_offset`, HBM→HBM with no host staging.
    /// Grows `dst`'s logical length. Errors out of bounds / over capacity.
    pub fn copy_to_peer(
        &self,
        src_offset: usize,
        dst: &DeviceSegment,
        dst_offset: usize,
        len: usize,
    ) -> Result<(), String> {
        if std::ptr::eq(self, dst) {
            return Err("peer copy to self is not supported".into());
        }
        let src_end = src_offset
            .checked_add(len)
            .ok_or("src offset+len overflow")?;
        let dst_end = dst_offset
            .checked_add(len)
            .ok_or("dst offset+len overflow")?;
        if dst_end > dst.capacity {
            return Err(format!(
                "peer copy exceeds dst HBM capacity ({dst_end} > {})",
                dst.capacity
            ));
        }
        // Distinct segments (distinct GPUs), guarded above, so locking src then
        // dst can't self-deadlock; a production path would order locks by address.
        let src_a = self.arena.lock().unwrap();
        if src_end > src_a.len {
            return Err("peer copy src out of segment bounds".into());
        }
        let mut dst_a = dst.arena.lock().unwrap();
        if len > 0 {
            let src_view = src_a.hbm.slice(src_offset..src_end);
            let mut dst_view = dst_a.hbm.slice_mut(dst_offset..dst_end);
            let (src_ptr, _src_sync) = src_view.device_ptr(&self.stream);
            let (dst_ptr, _dst_sync) = dst_view.device_ptr_mut(&dst.stream);
            // Synchronous wrt both contexts; the SyncOnDrop guards keep the slices
            // alive across the call.
            let res = unsafe {
                sys::cuMemcpyPeer(dst_ptr, dst.ctx.cu_ctx(), src_ptr, self.ctx.cu_ctx(), len)
            };
            if res != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemcpyPeer: {res:?}"));
            }
        }
        if dst_end > dst_a.len {
            dst_a.len = dst_end;
        }
        Ok(())
    }
}

/// Serve a GPU HBM [`DeviceSegment`] to peers over the same one-sided
/// `(offset, length)` wire as the host TCP segment, so an unmodified
/// [`crate::transport::tcp::TcpTransport`] peer READs / WRITEs GPU-resident
/// bytes. The bytes stage through `cudaMemcpy` here (GPUDirect would remove the
/// hop). The blocking copies run inline on the connection task — fine for the
/// reference path; a production server would `spawn_blocking` them.
pub async fn serve_device_segment(listener: TcpListener, segment: Arc<DeviceSegment>) {
    loop {
        let Ok((sock, _)) = listener.accept().await else {
            return;
        };
        let segment = segment.clone();
        tokio::spawn(async move {
            let _ = handle_conn(sock, segment).await;
        });
    }
}

async fn handle_conn(mut sock: TcpStream, segment: Arc<DeviceSegment>) -> std::io::Result<()> {
    loop {
        let op = match sock.read_u8().await {
            Ok(op) => op,
            Err(_) => return Ok(()), // peer closed
        };
        let offset = sock.read_u64().await? as usize;
        let len = sock.read_u64().await? as usize;
        match op {
            OP_READ => match segment.read(offset, len) {
                Ok(bytes) => {
                    sock.write_u8(ST_OK).await?;
                    sock.write_u64(bytes.len() as u64).await?;
                    sock.write_all(&bytes).await?;
                }
                Err(_) => sock.write_u8(ST_ERR).await?,
            },
            OP_WRITE => {
                let mut buf = vec![0u8; len];
                sock.read_exact(&mut buf).await?;
                match segment.write(offset, &buf) {
                    Ok(()) => sock.write_u8(ST_OK).await?,
                    Err(_) => sock.write_u8(ST_ERR).await?,
                }
            }
            _ => return Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::tcp::TcpTransport;
    use crate::transport::Transport;
    use bytes::Bytes;

    // Needs a real NVIDIA GPU; compiles everywhere (dynamic-loading) but only run
    // on a GPU box: `cargo test -p quillcache-transfer-engine --features cuda -- --ignored`.
    #[tokio::test]
    #[ignore = "requires an NVIDIA GPU"]
    async fn tcp_peer_reads_writes_a_gpu_hbm_segment() {
        // A GPU HBM segment served over the one-sided wire.
        let seg = DeviceSegment::new(0, 1 << 20).expect("alloc HBM segment");
        let off = seg.register(b"resident-in-HBM").expect("register"); // 15 bytes
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        tokio::spawn(serve_device_segment(listener, seg.clone()));

        // An UNMODIFIED TCP transport peer reads the GPU-resident bytes back.
        let tcp = TcpTransport;
        let got = tcp.read_remote(&endpoint, off, 15).await.expect("read HBM");
        assert_eq!(&got[..], b"resident-in-HBM");

        // ...and writes new bytes straight into HBM, which read back identically.
        tcp.write_remote(&endpoint, 100, Bytes::from_static(b"written-into-HBM"))
            .await
            .expect("write HBM");
        let got2 = tcp
            .read_remote(&endpoint, 100, 16)
            .await
            .expect("read HBM 2");
        assert_eq!(&got2[..], b"written-into-HBM");
    }

    // Needs TWO NVIDIA GPUs. Zero-copy HBM→HBM peer copy: bytes registered on
    // GPU 0's segment land in GPU 1's segment with no host staging, and read back
    // identically. Run on a 2-GPU box:
    // `cargo test -p quillcache-transfer-engine --features cuda -- --ignored`.
    #[test]
    #[ignore = "requires two NVIDIA GPUs"]
    fn peer_copy_moves_bytes_gpu0_to_gpu1() {
        let src = DeviceSegment::new(0, 1 << 20).expect("alloc HBM segment on GPU 0");
        let dst = DeviceSegment::new(1, 1 << 20).expect("alloc HBM segment on GPU 1");
        let payload = b"zero-copy-across-GPUs";
        let off = src.register(payload).expect("register on GPU 0");

        // Best-effort enable the P2P fast path; the copy is correct either way.
        if src.can_access_peer(&dst).unwrap_or(false) {
            src.enable_peer_access(&dst).expect("enable peer access");
        }
        src.copy_to_peer(off as usize, &dst, 0, payload.len())
            .expect("HBM→HBM peer copy");

        // Read it back off GPU 1 (D2H) — the bytes crossed GPU0→GPU1 directly.
        let got = dst.read(0, payload.len()).expect("read GPU 1 segment");
        assert_eq!(&got[..], payload);
    }

    // Needs TWO NVIDIA GPUs (ideally NVLink-connected). Measures the zero-copy
    // win: HBM→HBM peer copy vs the host-staged path (D2H then H2D) for the same
    // bytes. Prints a `QC-P2P ...` line with both bandwidths + the speedup.
    #[test]
    #[ignore = "requires two NVIDIA GPUs"]
    fn peer_copy_bandwidth_vs_host_staged() {
        use std::time::Instant;
        let bytes = 256 * 1024 * 1024; // 256 MiB
        let src = DeviceSegment::new(0, bytes).expect("src segment GPU 0");
        let dst = DeviceSegment::new(1, bytes).expect("dst segment GPU 1");
        src.register(&vec![7u8; bytes]).expect("fill src HBM");

        let p2p = src.can_access_peer(&dst).unwrap_or(false);
        if p2p {
            src.enable_peer_access(&dst).expect("enable peer access");
        }

        let iters = 10;
        src.copy_to_peer(0, &dst, 0, bytes).expect("warm peer copy"); // warm up
        let t = Instant::now();
        for _ in 0..iters {
            src.copy_to_peer(0, &dst, 0, bytes).expect("peer copy");
        }
        let peer_gbs = (bytes as f64 * iters as f64) / t.elapsed().as_secs_f64() / 1e9;

        // The host-bounced path: D2H off GPU 0, then H2D onto GPU 1.
        let t = Instant::now();
        for _ in 0..iters {
            let host = src.read(0, bytes).expect("d2h");
            dst.write(0, &host).expect("h2d");
        }
        let staged_gbs = (bytes as f64 * iters as f64) / t.elapsed().as_secs_f64() / 1e9;

        eprintln!(
            "QC-P2P p2p_enabled={p2p} peer={peer_gbs:.1}GB/s staged={staged_gbs:.1}GB/s speedup={:.2}x",
            peer_gbs / staged_gbs.max(f64::MIN_POSITIVE)
        );
        assert!(peer_gbs > 0.0, "peer-copy bandwidth must be positive");
    }
}
