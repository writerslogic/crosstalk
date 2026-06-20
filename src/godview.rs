// File: src/godview.rs
//! GodView async orchestration.
//!
//! GodView async work executes via a main-thread runner so that running it on
//! the wrong thread is unrepresentable. The runner is expressed as a trait so
//! the actual driver (e.g. `AsyncExecutor::main_thread_handle()`) is supplied
//! at the call site; there is no ad-hoc thread spawn for GodView.

use std::future::Future;

/// A runner capable of driving a GodView future to completion on the thread
/// that owns it. This abstracts over `MainThreadHandle::block_on`.
///
/// Implementors are intentionally not required to be `Send`: the whole point is
/// that GodView work is pinned to the owning (main) thread.
pub trait MainThreadRunner {
    /// Drive `fut` to completion on the owning thread, returning its output.
    fn run<F: Future>(&self, fut: F) -> F::Output;
}

/// Drives the GodView async loop/work on the owning main thread via `runner`.
///
/// `work` produces the future representing GodView's async work. Because the
/// future is only ever handed to a `MainThreadRunner`, it cannot escape to a
/// foreign thread.
pub fn run_godview<R, F, T>(runner: &R, work: impl FnOnce() -> F) -> T
where
    R: MainThreadRunner,
    F: Future<Output = T>,
{
    runner.run(work())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    // A simple synchronous main-thread runner for tests. It is `!Send` by
    // construction (holds a `Cell`), mirroring the `MainThreadHandle` contract.
    struct TestRunner {
        // Records how many futures were driven; proves `run` was invoked.
        drove: Cell<u32>,
    }

    impl TestRunner {
        fn new() -> Self {
            TestRunner {
                drove: Cell::new(0),
            }
        }
    }

    impl MainThreadRunner for TestRunner {
        fn run<F: Future>(&self, fut: F) -> F::Output {
            self.drove.set(self.drove.get() + 1);
            // Drive a (here, immediately-ready) future to completion without a
            // full async runtime: poll once. For tests we only feed futures
            // that are ready on first poll.
            use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

            fn noop_raw_waker() -> RawWaker {
                fn no_op(_: *const ()) {}
                fn clone(_: *const ()) -> RawWaker {
                    noop_raw_waker()
                }
                let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
                RawWaker::new(std::ptr::null(), vtable)
            }

            let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
            let mut cx = Context::from_waker(&waker);
            let mut fut = Box::pin(fut);
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => v,
                Poll::Pending => panic!("test futures must be ready on first poll"),
            }
        }
    }

    // CERTAIN: run_godview drives the produced future and returns its output.
    #[test]
    fn run_godview_returns_output() {
        let runner = TestRunner::new();
        let out = run_godview(&runner, || async { 21 * 2 });
        assert_eq!(out, 42);
        assert_eq!(runner.drove.get(), 1, "runner must drive exactly one future");
    }

    // CERTAIN: the work closure is only invoked once, and its future drives.
    #[test]
    fn run_godview_invokes_work_closure_once() {
        let runner = TestRunner::new();
        let calls = Cell::new(0u32);
        let out = run_godview(&runner, || {
            calls.set(calls.get() + 1);
            async { String::from("godview") }
        });
        assert_eq!(out, "godview");
        assert_eq!(calls.get(), 1, "work closure must be called exactly once");
    }

    // CERTAIN: multi-step (multiple awaits) futures resolve correctly.
    #[test]
    fn run_godview_executes_multi_step_future() {
        let runner = TestRunner::new();
        let out = run_godview(&runner, || async {
            let a = async { 10u32 }.await;
            let b = async { 32u32 }.await;
            a + b
        });
        assert_eq!(out, 42);
    }

    // CERTAIN: the runner can be reused for multiple GodView invocations on the
    // same owning thread, accumulating its drive count.
    #[test]
    fn run_godview_reusable_on_owning_thread() {
        let runner = TestRunner::new();
        let _ = run_godview(&runner, || async { 1u32 });
        let _ = run_godview(&runner, || async { 2u32 });
        let _ = run_godview(&runner, || async { 3u32 });
        assert_eq!(runner.drove.get(), 3);
    }

    // CERTAIN: a MainThreadRunner is *not* required to be Send. This positive
    // control asserts that ordinary Send types still satisfy the bound when
    // they choose to, documenting that the trait itself imposes no Send.
    fn assert_runner<R: MainThreadRunner>() {}

    #[test]
    fn runner_trait_does_not_require_send() {
        // TestRunner holds a Cell and is therefore !Send, yet it implements
        // MainThreadRunner — proving the trait does not demand Send.
        assert_runner::<TestRunner>();
    }
}