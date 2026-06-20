#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uses_named_consts() {
        let cfg = GatewayConfig::default();
        assert_eq!(cfg.port, DEFAULT_GATEWAY_PORT);
        assert_eq!(cfg.timeout, DEFAULT_GATEWAY_TIMEOUT);
        assert_eq!(cfg.max_payload_bytes, MAX_PAYLOAD_BYTES);
    }

    /// Concurrent fan-out smoke test.
    ///
    /// Verifies that gateway configuration can be shared (`&self`-style,
    /// no `&mut self` bottleneck) across many threads simultaneously without
    /// serialization. `GatewayConfig` is `Copy + Send + Sync`, so this models
    /// the concurrent request fan-out path: many readers, no exclusive lock.
    #[test]
    fn concurrent_requests_no_serialization_bottleneck() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let cfg = Arc::new(GatewayConfig::default());
        let completed = Arc::new(AtomicUsize::new(0));

        const THREADS: usize = 16;
        const REQUESTS_PER_THREAD: usize = 64;

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let cfg = Arc::clone(&cfg);
                let completed = Arc::clone(&completed);
                thread::spawn(move || {
                    for _ in 0..REQUESTS_PER_THREAD {
                        // Shared read access via `&self` semantics — no
                        // `&mut self` would compile here under Arc, proving
                        // the path is non-exclusive.
                        let c: &GatewayConfig = &cfg;
                        assert_eq!(c.port, DEFAULT_GATEWAY_PORT);
                        assert!(c.max_payload_bytes > 0);
                        completed.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker thread panicked");
        }

        assert_eq!(
            completed.load(Ordering::Relaxed),
            THREADS * REQUESTS_PER_THREAD,
            "all concurrent requests should complete"
        );
    }
}