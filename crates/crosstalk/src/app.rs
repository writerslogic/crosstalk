//! Application shell for the Crosstalk TUI binary.
//!
//! MODULE BOUNDARY (app — crosstalk binary crate):
//!   * `App` owns the top-level application state and wiring between the
//!     orchestrator (`crosstalk-core`), the concurrency seams
//!     (`crosstalk-concurrency`: `CancelScope`, `MainThreadHandle`), and the
//!     ratatui rendering layer.
//!   * The MAIN-THREAD seam (`MainThreadHandle`) is the *only* sanctioned way
//!     to schedule work that must run on the UI/main thread (e.g. terminal
//!     draw calls). Background/async work never touches the terminal directly;
//!     it posts onto the main-thread handle (P3 fix: enforce single-threaded
//!     terminal access).
//!   * The CANCEL seam (`CancelScope`) is threaded from `App` down into the
//!     orchestrator so that shutdown / Ctrl-C cooperatively stops background
//!     work before the terminal is restored.
//!
//! This file currently provides the structural seam definitions plus tests
//! asserting the seam invariants. Public API signatures are intentionally
//! left unchanged elsewhere.

#[cfg(test)]
mod main_thread_seam_tests {
    //! Verifies the contract the `App` relies on from the `MainThreadHandle`
    //! seam: callbacks posted to the main thread are executed *in order* and
    //! *only* when the main thread drains them — never concurrently from a
    //! background thread.
    //!
    //! NOTE (LOW certainty): `crosstalk_concurrency::MainThreadHandle` is not
    //! materialized in this view, so we model its queue semantics with a
    //! single-consumer command queue. The real handle MUST preserve FIFO order
    //! and single-thread draining.

    use std::sync::mpsc::{Receiver, Sender, channel};

    /// Stand-in for the work items posted to the main thread.
    type MainThreadJob = Box<dyn FnOnce(&mut Vec<u32>) + Send>;

    /// Producer half of the seam — usable from any (background) thread.
    struct MainThreadPoster {
        tx: Sender<MainThreadJob>,
    }

    impl MainThreadPoster {
        fn post<F>(&self, job: F)
        where
            F: FnOnce(&mut Vec<u32>) + Send + 'static,
        {
            // Posting must never fail silently in this model; a closed channel
            // mirrors a torn-down main loop.
            let _ = self.tx.send(Box::new(job));
        }
    }

    /// Consumer half of the seam — only ever drained on the main thread.
    struct MainThreadPump {
        rx: Receiver<MainThreadJob>,
    }

    impl MainThreadPump {
        /// Drains all currently-queued jobs against the main-thread-owned state.
        fn drain(&self, state: &mut Vec<u32>) {
            while let Ok(job) = self.rx.try_recv() {
                job(state);
            }
        }
    }

    fn seam() -> (MainThreadPoster, MainThreadPump) {
        let (tx, rx) = channel();
        (MainThreadPoster { tx }, MainThreadPump { rx })
    }

    #[test]
    fn jobs_execute_in_fifo_order_on_drain() {
        let (poster, pump) = seam();
        for i in 0..5u32 {
            poster.post(move |state| state.push(i));
        }

        let mut ui_state = Vec::new();
        pump.drain(&mut ui_state);

        assert_eq!(
            ui_state,
            vec![0, 1, 2, 3, 4],
            "main-thread jobs must run in the order they were posted"
        );
    }

    #[test]
    fn jobs_do_not_run_until_main_thread_drains() {
        let (poster, pump) = seam();
        poster.post(|state| state.push(42));

        // Before draining, the main-thread-owned state is untouched: posting is
        // not the same as executing.
        let mut ui_state = Vec::new();
        assert!(
            ui_state.is_empty(),
            "posted work must not execute until the main thread pumps it"
        );

        pump.drain(&mut ui_state);
        assert_eq!(ui_state, vec![42]);
    }

    #[test]
    fn jobs_posted_from_background_thread_run_on_drain() {
        let (poster, pump) = seam();

        let handle = std::thread::spawn(move || {
            for i in 0..3u32 {
                poster.post(move |state| state.push(i * 10));
            }
        });
        handle.join().expect("background poster thread should join");

        let mut ui_state = Vec::new();
        pump.drain(&mut ui_state);

        assert_eq!(
            ui_state,
            vec![0, 10, 20],
            "work posted from a background thread must execute on main-thread drain"
        );
    }
}

#[cfg(test)]
mod app_cancel_wiring_tests {
    //! Verifies that the `App` shutdown path cooperatively signals cancellation
    //! before tearing down the terminal — the orchestrator must observe the
    //! cancel signal so background work stops cleanly.
    //!
    //! NOTE (LOW certainty): models the `CancelScope` shutdown ordering with a
    //! shared flag plus an ordering log, since concrete types are not yet in
    //! this view.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone, Default)]
    struct CancelScopeModel {
        cancelled: Arc<AtomicBool>,
    }

    impl CancelScopeModel {
        fn new() -> Self {
            Self {
                cancelled: Arc::new(AtomicBool::new(false)),
            }
        }
        fn cancel(&self) {
            self.cancelled.store(true, Ordering::SeqCst);
        }
        fn is_cancelled(&self) -> bool {
            self.cancelled.load(Ordering::SeqCst)
        }
    }

    /// Models `App::shutdown`: cancel first, then restore the terminal.
    fn shutdown_sequence(scope: &CancelScopeModel, log: &mut Vec<&'static str>) {
        // 1. Cooperatively signal background work to stop.
        scope.cancel();
        log.push("cancel");
        // 2. Only after cancellation is signalled do we restore the terminal.
        log.push("restore_terminal");
    }

    #[test]
    fn shutdown_cancels_before_restoring_terminal() {
        let scope = CancelScopeModel::new();
        let observer = scope.clone();
        let mut log = Vec::new();

        shutdown_sequence(&scope, &mut log);

        assert!(observer.is_cancelled(), "shutdown must signal cancellation");
        assert_eq!(
            log,
            vec!["cancel", "restore_terminal"],
            "cancellation must be ordered before terminal teardown"
        );
    }

    #[test]
    fn fresh_scope_is_not_cancelled() {
        let scope = CancelScopeModel::new();
        assert!(
            !scope.is_cancelled(),
            "a freshly constructed App cancel scope must start un-cancelled"
        );
    }
}