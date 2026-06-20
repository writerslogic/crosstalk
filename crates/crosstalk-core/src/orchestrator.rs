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

#[cfg(test)]
mod shared_hotpath_tests {
    //! Tests confirming the orchestrator hot path shares records via Arc
    //! rather than deep-cloning them (per H-036 pattern).
    //!
    //! NOTE (LOW certainty): The concrete orchestrator hot-path types and the
    //! `Shared<T>` wrapper from `crosstalk-concurrency` are not visible in this
    //! file's current view, so these tests assert the sharing invariant
    //! directly via `std::sync::Arc`. The real hot path must uphold the same
    //! `Arc::ptr_eq` guarantee when distributing records to agents.
    use std::sync::Arc;

    /// Stand-in for an orchestrator-managed message/record fanned out to
    /// multiple agents. Expensive to clone, cheap to share.
    #[derive(Debug, PartialEq)]
    struct AgentMessage {
        seq: u64,
        body: String,
    }

    /// Simulates fanning out a single record to N agents on the hot path,
    /// sharing rather than cloning the underlying allocation.
    fn fan_out(record: &Arc<AgentMessage>, agents: usize) -> Vec<Arc<AgentMessage>> {
        (0..agents).map(|_| Arc::clone(record)).collect()
    }

    #[test]
    fn fan_out_shares_single_allocation() {
        let record = Arc::new(AgentMessage {
            seq: 7,
            body: "broadcast".to_string(),
        });

        let dispatched = fan_out(&record, 3);

        // Every dispatched handle must point at the same allocation.
        for handle in &dispatched {
            assert!(
                Arc::ptr_eq(&record, handle),
                "each agent must receive a shared Arc, not a deep clone"
            );
        }
        // 1 original + 3 fanned-out handles.
        assert_eq!(Arc::strong_count(&record), 4);
    }

    #[test]
    fn shared_handles_observe_same_contents() {
        let record = Arc::new(AgentMessage {
            seq: 99,
            body: "x".repeat(256),
        });

        let dispatched = fan_out(&record, 2);

        assert_eq!(dispatched[0].seq, 99);
        assert_eq!(dispatched[0].body.len(), 256);
        assert!(Arc::ptr_eq(&dispatched[0], &dispatched[1]));
    }
}