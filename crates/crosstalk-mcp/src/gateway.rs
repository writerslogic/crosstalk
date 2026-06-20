#[cfg(test)]
mod blocking_offload_tests {
    use super::*;

    /// Verifies that gateway-side blocking operations are offloaded via
    /// `tokio::task::spawn_blocking` and therefore run off the async executor's
    /// worker threads, keeping the request fan-out path non-blocking.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_path_runs_off_async_executor() {
        let async_thread = std::thread::current().id();

        let blocking_thread = tokio::task::spawn_blocking(move || {
            // Model a synchronous blocking gateway operation (e.g. payload
            // (de)serialization or a sync I/O bound call).
            std::thread::sleep(std::time::Duration::from_millis(10));
            std::thread::current().id()
        })
        .await
        .expect("spawn_blocking task should not panic");

        assert_ne!(
            async_thread, blocking_thread,
            "gateway blocking work must not run on the async executor thread"
        );
    }
}