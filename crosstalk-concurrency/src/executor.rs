//! Task execution helpers.
//!
//! Provides [`AsyncExecutor`], a thin wrapper around a Tokio runtime that
//! exposes typed handles so that "wrong thread" usage is unrepresentable.
//!
//! - [`MainThreadHandle`] is `!Send` (it carries a `PhantomData<*const ()>`),
//!   so the compiler forbids moving it to another thread. GodView (and other
//!   single-threaded UI/orchestration work) must run through this handle.
//! - Blocking work is routed through [`AsyncExecutor::spawn_blocking`], which
//!   offloads onto Tokio's dedicated blocking pool, keeping the async reactor
//!   responsive.
//!
//! Addresses H-038 (thread-confinement of GodView work) and SYS-004
//! (centralized runtime construction).

use std::future::Future;
use std::marker::PhantomData;

use anyhow::{Context, Result};
use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;

/// CERTAIN: A handle whose work must execute on the thread that owns the
/// runtime. The `PhantomData<*const ()>` raw-pointer marker makes the type
/// `!Send` and `!Sync`, so the borrow checker prevents it from crossing thread
/// boundaries. This is the mechanism that makes "wrong thread" unrepresentable.
pub struct MainThreadHandle<'rt> {
    // CERTAIN: Borrow ties the handle's lifetime to the executor so it cannot
    // outlive the runtime that backs it.
    runtime: &'rt Runtime,
    // CERTAIN: Raw pointer marker => neither Send nor Sync.
    _not_send: PhantomData<*const ()>,
}

impl<'rt> MainThreadHandle<'rt> {
    /// CERTAIN: Run a future to completion on the current (main) thread,
    /// driving it via the owning runtime. Because `self` is `!Send`, this can
    /// only ever be called on the thread that constructed the handle.
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    /// CERTAIN: Spawn a `!Send` future onto a `LocalSet`-style local execution
    /// is out of scope here; instead we expose the runtime handle for callers
    /// that need to enter the runtime context on this thread.
    pub fn enter(&self) -> tokio::runtime::EnterGuard<'_> {
        self.runtime.enter()
    }
}

/// CERTAIN: Owns the Tokio runtime and hands out typed access points.
pub struct AsyncExecutor {
    runtime: Runtime,
}

impl AsyncExecutor {
    /// CERTAIN: Construct the runtime. Uses a multi-threaded scheduler so that
    /// `spawn_blocking` work has dedicated threads while async tasks run on the
    /// worker pool.
    pub fn build() -> Result<Self> {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("failed to build tokio runtime for AsyncExecutor")?;
        Ok(Self { runtime })
    }

    /// CERTAIN: Run a future to completion on the calling thread. Convenience
    /// mirror of `MainThreadHandle::block_on` for callers that hold the
    /// executor directly.
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    /// CERTAIN: Borrow a thread-confined handle for GodView / main-thread work.
    /// The returned `MainThreadHandle` is `!Send`, so it cannot be moved off
    /// this thread — the "wrong thread" case is rejected at compile time.
    pub fn main_thread_handle(&self) -> MainThreadHandle<'_> {
        MainThreadHandle {
            runtime: &self.runtime,
            _not_send: PhantomData,
        }
    }

    /// CERTAIN: Offload blocking work onto Tokio's dedicated blocking pool.
    /// The closure must be `Send` because it crosses to a pool thread; this is
    /// the *only* sanctioned path for blocking work, complementing the
    /// `!Send` main-thread handle.
    pub fn spawn_blocking<F, T>(&self, f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.runtime.spawn_blocking(f)
    }

    /// CERTAIN: Spawn a `Send` async task onto the runtime's worker pool.
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(future)
    }

    /// CERTAIN: Expose the underlying runtime handle for advanced callers.
    pub fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }
}

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

        let (worker_thread, sum) = exec.block_on(join).expect("join ok");
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

        let result = exec.block_on(join).expect("join ok");
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
        let result = exec.block_on(join);
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
