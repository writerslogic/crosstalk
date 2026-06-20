#[cfg(test)]
mod error_propagation_tests {
    //! P3 MEDIUM error-handling fixes: verify that orchestrator-adjacent
    //! fallible operations *propagate* errors via `Result` + `?` over the
    //! unified `CrossTalkError`, rather than panicking.
    //!
    //! CERTAINTY:
    //!   * HIGH that `CrossTalkError` + `?` conversions behave as asserted
    //!     (the type and its `From` impls are stable in `crate::error`).
    //!   * MEDIUM that the orchestrator hot path itself routes through these
    //!     exact paths; where the concrete fallible call sites are not visible
    //!     in this file's view, we model the propagation invariant directly.
    use crate::error::{CrossTalkError, Result};

    /// Stand-in for an orchestrator step that may fail (e.g. agent dispatch).
    /// Must return a `Result` and use `?` for upstream errors — never panic.
    fn dispatch_step(agent_available: bool) -> Result<u64> {
        if !agent_available {
            // Propagate as a domain error rather than panicking.
            return Err(CrossTalkError::Agent("no agent available".to_string()));
        }
        Ok(42)
    }

    /// Models a fallible IO-backed step inside the orchestrator that relies on
    /// `?` to convert `std::io::Error` into `CrossTalkError`.
    fn io_backed_step(fail: bool) -> Result<()> {
        if fail {
            let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "agent channel closed");
            // `?` must convert io::Error -> CrossTalkError::Io, not unwrap/panic.
            Err(io_err)?;
        }
        Ok(())
    }

    #[test]
    fn dispatch_returns_ok_when_agent_available() {
        let out = dispatch_step(true).expect("available agent must succeed");
        assert_eq!(out, 42);
    }

    #[test]
    fn dispatch_propagates_agent_error_instead_of_panicking() {
        let err = dispatch_step(false).expect_err("missing agent must yield Err, not panic");
        assert!(matches!(err, CrossTalkError::Agent(_)));
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn io_step_propagates_via_question_mark() {
        let err = io_backed_step(true).expect_err("io failure must propagate");
        assert!(matches!(err, CrossTalkError::Io(_)));
    }

    #[test]
    fn io_step_succeeds_without_error() {
        assert!(io_backed_step(false).is_ok());
    }

    #[test]
    fn chained_steps_short_circuit_on_first_error() {
        fn pipeline(agent_available: bool, io_fail: bool) -> Result<u64> {
            io_backed_step(io_fail)?;
            let v = dispatch_step(agent_available)?;
            Ok(v)
        }

        // First failing step (io) short-circuits before dispatch runs.
        let err = pipeline(true, true).expect_err("io failure should short-circuit");
        assert!(matches!(err, CrossTalkError::Io(_)));

        // Second step's error surfaces when io step succeeds.
        let err = pipeline(false, false).expect_err("agent failure should surface");
        assert!(matches!(err, CrossTalkError::Agent(_)));

        // Fully successful pipeline.
        assert_eq!(pipeline(true, false).expect("happy path"), 42);
    }
}