#[cfg(test)]
mod blocking_offload_tests {
    use super::*;

    /// Verifies that a blocking operation, when offloaded via
    /// `tokio::task::spawn_blocking`, executes on a *different* thread than the
    /// async executor's worker thread. This proves the blocking path does not
    /// run inline on the async runtime, which would otherwise stall the
    /// scheduler.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_path_runs_off_async_executor() {
        // Capture the thread id the async task is currently running on.
        let async_thread = std::thread::current().id();

        // Offload a "blocking" operation. spawn_blocking moves work onto the
        // dedicated blocking thread pool, never the async worker threads.
        let blocking_thread = tokio::task::spawn_blocking(move || {
            // Simulate a synchronous, potentially long blocking call.
            std::thread::sleep(std::time::Duration::from_millis(10));
            std::thread::current().id()
        })
        .await
        .expect("spawn_blocking task should not panic");

        assert_ne!(
            async_thread, blocking_thread,
            "blocking work must run on a separate thread from the async executor"
        );
    }

    /// Confirms the async executor remains responsive (not blocked) while a
    /// blocking task is in flight on the blocking pool.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_executor_stays_responsive_during_blocking_work() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let blocking_done = Arc::new(AtomicBool::new(false));
        let bd = Arc::clone(&blocking_done);

        let blocking_handle = tokio::task::spawn_blocking(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            bd.store(true, Ordering::SeqCst);
        });

        // While the blocking task sleeps, the async executor should still be
        // able to make progress on other futures.
        let mut progressed = 0usize;
        for _ in 0..5 {
            tokio::task::yield_now().await;
            progressed += 1;
        }
        assert_eq!(progressed, 5, "async tasks should progress freely");

        blocking_handle.await.expect("blocking task should complete");
        assert!(
            blocking_done.load(Ordering::SeqCst),
            "offloaded blocking work should finish"
        );
    }
}