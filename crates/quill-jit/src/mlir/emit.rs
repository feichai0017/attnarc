use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_symbol(prefix: &str) -> String {
    let id = NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{id}")
}
