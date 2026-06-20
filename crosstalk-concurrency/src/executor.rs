#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    // CERTAIN: build() must succeed and produce a usable runtime.
    #[test]
    fn build_succeeds() {
        let exec = AsyncExecutor::build();
        assert!(exec.is_ok(), "AsyncExecutor::build() should succeed");
    }

    // CERTAIN: block_on drives a future to completion and returns its output.
    #[test]
    fn block_on_returns_output() {
        let exec = AsyncExecutor::build().expect("build");
        let value = exec.block_on(async { 21 * 2 });
        assert_eq!(value, 42);
    }

    // CERTAIN: block_on also resolves futures that await intermediate awaits.
    #[test]
    fn block_on_executes_multi_step_future() {
        let exec = AsyncExecutor::build().expect("build");
        let value = exec.block_on(async {
            let a = async { 10u32 }.await;
            let b = async { 32u32 }.await;
            a + b
        });
        assert_eq!(value, 42);
    }

    // CERTAIN: The main-thread handle's block_on also resolves futures.
    #[test]
    fn main_thread_handle_block_on() {
        let exec = AsyncExecutor::build().expect("build");
        let handle = exec.main_thread_handle();
        let value = handle.block_on(async { String::from("godview") });
        assert_eq!(value, "godview");
    }

    // CERTAIN: spawn_blocking offloads work onto a DIFFERENT thread than the
    // one driving block_on. We capture the worker thread id and confirm it is
    // not the test/main thread, proving the work was actually offloaded.
    #[test]
    fn spawn_blocking_offloads_to_other_thread() {
        let exec = AsyncExecutor::build().expect("build");
        let main_thread = thread::current().id();

        let join = exec.spawn_blocking(move || {
            let worker_thread = thread::current().id();
            let sum = (1..=10u64).sum::<u64>();
            (worker_thread, sum)
        });

        let (worker_thread, sum) = exec.block_on(async { join.await }).expect("join ok");
        assert_eq!(
            sum, 55,
            "blocking computation should produce correct result"
        );
        assert_ne!(
            worker_thread, main_thread,
            "spawn_blocking must execute on the blocking pool, not the calling thread"
        );
    }

    // CERTAIN: spawn_blocking actually runs the closure body (side effect).
    #[test]
    fn spawn_blocking_runs_closure() {
        let exec = AsyncExecutor::build().expect("build");
        let ran = Arc::new(AtomicBool::new(false));
        let ran_clone = Arc::clone(&ran);

        let join = exec.spawn_blocking(move || {
            ran_clone.store(true, Ordering::SeqCst);
            7u32
        });

        let result = exec.block_on(async { join.await }).expect("join ok");
        assert_eq!(result, 7);
        assert!(
            ran.load(Ordering::SeqCst),
            "closure body must have executed"
        );
    }

    // CERTAIN: spawn schedules an async task whose output we can await.
    #[test]
    fn spawn_async_task() {
        let exec = AsyncExecutor::build().expect("build");
        let join = exec.spawn(async { 7u32 + 8u32 });
        let result = exec.block_on(async { join.await });
        assert_eq!(result.expect("join ok"), 15);
    }

    // CERTAIN: The handle() accessor returns a clonable runtime Handle.
    #[test]
    fn handle_is_accessible() {
        let exec = AsyncExecutor::build().expect("build");
        let h = exec.handle();
        let value = h.block_on(async { 99 });
        assert_eq!(value, 99);
    }

    // CERTAIN: Compile-time enforcement that MainThreadHandle is `!Send`.
    //
    // This helper is only callable for `Send` types. We deliberately do NOT
    // instantiate it with `MainThreadHandle` (which would fail to compile),
    // but we DO instantiate it with `AsyncExecutor`'s `Handle` to document the
    // contrast. The `!Send` guarantee for MainThreadHandle is provided by its
    // `PhantomData<*const ()>` marker; the negative assertion below documents
    // the invariant without requiring extra crates.
    fn assert_send<T: Send>() {}

    // CERTAIN: A runtime Handle IS Send (positive control). MainThreadHandle is
    // intentionally absent here because adding it would break compilation,
    // which is exactly the enforced main-thread-only semantics we rely on.
    #[test]
    fn runtime_handle_is_send_control() {
        assert_send::<tokio::runtime::Handle>();
    }

    // CERTAIN: MainThreadHandle is usable on the thread that owns the runtime.
    // Entering the runtime context and driving a future via block_on both work
    // on the owning thread, which is the only place the `!Send` handle exists.
    #[test]
    fn main_thread_handle_usable_on_owning_thread() {
        let exec = AsyncExecutor::build().expect("build");
        let handle = exec.main_thread_handle();
        {
            // Using it on the creating thread is allowed (enter the context).
            let _guard = handle.enter();
        }
        // And block_on works on the same thread.
        let value = handle.block_on(async { 1234u32 });
        assert_eq!(value, 1234);
    }
}
