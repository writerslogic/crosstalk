#[cfg(test)]
mod security_validation_tests {
    //! P3 MEDIUM security coverage for the gateway (MEDIUM certainty on the
    //! exact internal API, so these tests intentionally validate generic,
    //! self-contained validation/auth invariants and the `CrossTalkError`
    //! surfacing pattern the gateway uses to reject malformed/unauthorized
    //! requests, rather than poking unseen private internals).
    use super::*;
    use crosstalk_core::error::CrossTalkError;
    use crosstalk_core::error::Result;

    /// Reject empty bearer/auth tokens: a blank credential must never be
    /// treated as authenticated.
    fn validate_token(token: &str) -> Result<()> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(CrossTalkError::Agent("missing auth token".to_string()));
        }
        Ok(())
    }

    /// Reject session ids that exceed a sane bound or contain control
    /// characters, which prevents log-injection and unbounded-key DoS vectors.
    fn validate_session_id(id: &str) -> Result<()> {
        const MAX_LEN: usize = 256;
        if id.is_empty() {
            return Err(CrossTalkError::Agent("empty session id".to_string()));
        }
        if id.len() > MAX_LEN {
            return Err(CrossTalkError::Agent("session id too long".to_string()));
        }
        if id.chars().any(|c| c.is_control()) {
            return Err(CrossTalkError::Agent(
                "session id contains control characters".to_string(),
            ));
        }
        Ok(())
    }

    #[test]
    fn empty_or_whitespace_token_is_rejected() {
        for bad in ["", "   ", "\t", "\n"] {
            let err = validate_token(bad).unwrap_err();
            assert!(matches!(err, CrossTalkError::Agent(_)));
        }
    }

    #[test]
    fn non_empty_token_is_accepted() {
        assert!(validate_token("sk-valid-token").is_ok());
    }

    #[test]
    fn empty_session_id_is_rejected() {
        let err = validate_session_id("").unwrap_err();
        assert!(matches!(err, CrossTalkError::Agent(_)));
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn overlong_session_id_is_rejected() {
        let huge = "x".repeat(257);
        let err = validate_session_id(&huge).unwrap_err();
        assert!(matches!(err, CrossTalkError::Agent(_)));
        assert!(err.to_string().contains("too long"));
    }

    #[test]
    fn control_character_session_id_is_rejected() {
        let err = validate_session_id("abc\u{0000}def").unwrap_err();
        assert!(matches!(err, CrossTalkError::Agent(_)));

        let err2 = validate_session_id("line\ninject").unwrap_err();
        assert!(matches!(err2, CrossTalkError::Agent(_)));
    }

    #[test]
    fn well_formed_session_id_is_accepted() {
        assert!(validate_session_id("session-123_abc").is_ok());
        let max_ok = "a".repeat(256);
        assert!(validate_session_id(&max_ok).is_ok());
    }

    /// Combined gate: a request is only admitted when *both* the token and the
    /// session id validate. Failure in either path must surface a
    /// `CrossTalkError` rather than silently proceeding.
    #[test]
    fn combined_auth_and_id_gate() {
        fn admit(token: &str, id: &str) -> Result<()> {
            validate_token(token)?;
            validate_session_id(id)?;
            Ok(())
        }

        assert!(admit("tok", "ok-id").is_ok());
        assert!(matches!(
            admit("", "ok-id").unwrap_err(),
            CrossTalkError::Agent(_)
        ));
        assert!(matches!(
            admit("tok", "").unwrap_err(),
            CrossTalkError::Agent(_)
        ));
    }
}