use std::error::Error as StdError;
use std::fmt;

/// The unified error type for Crosstalk, used for `?` propagation across crates.
///
/// Implements [`std::error::Error`] and provides `From` conversions for common
/// error sources without introducing additional dependencies.
#[derive(Debug)]
#[non_exhaustive]
pub enum CrossTalkError {
    /// An I/O error from the standard library.
    Io(std::io::Error),
    /// A formatting error.
    Fmt(std::fmt::Error),
    /// A configuration-related error with a descriptive message.
    Config(String),
    /// An agent orchestration error with a descriptive message.
    Agent(String),
    /// A generic, boxed error from another source.
    Other(Box<dyn StdError + Send + Sync + 'static>),
}

impl fmt::Display for CrossTalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrossTalkError::Io(e) => write!(f, "I/O error: {e}"),
            CrossTalkError::Fmt(e) => write!(f, "formatting error: {e}"),
            CrossTalkError::Config(msg) => write!(f, "configuration error: {msg}"),
            CrossTalkError::Agent(msg) => write!(f, "agent error: {msg}"),
            CrossTalkError::Other(e) => write!(f, "error: {e}"),
        }
    }
}

impl StdError for CrossTalkError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            CrossTalkError::Io(e) => Some(e),
            CrossTalkError::Fmt(e) => Some(e),
            CrossTalkError::Config(_) => None,
            CrossTalkError::Agent(_) => None,
            CrossTalkError::Other(e) => Some(e.as_ref()),
        }
    }
}

impl From<std::io::Error> for CrossTalkError {
    fn from(e: std::io::Error) -> Self {
        CrossTalkError::Io(e)
    }
}

impl From<std::fmt::Error> for CrossTalkError {
    fn from(e: std::fmt::Error) -> Self {
        CrossTalkError::Fmt(e)
    }
}

impl From<Box<dyn StdError + Send + Sync + 'static>> for CrossTalkError {
    fn from(e: Box<dyn StdError + Send + Sync + 'static>) -> Self {
        CrossTalkError::Other(e)
    }
}

impl From<String> for CrossTalkError {
    fn from(msg: String) -> Self {
        CrossTalkError::Config(msg)
    }
}

impl From<&str> for CrossTalkError {
    fn from(msg: &str) -> Self {
        CrossTalkError::Config(msg.to_string())
    }
}

/// Convenient result alias used across Crosstalk crates.
pub type Result<T, E = CrossTalkError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn io_conversion_via_question_mark_compiles() {
        fn inner() -> Result<()> {
            let io_err = std::io::Error::new(std::io::ErrorKind::Other, "boom");
            Err(io_err)?;
            Ok(())
        }
        let err = inner().unwrap_err();
        assert!(matches!(err, CrossTalkError::Io(_)));
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn fmt_conversion_compiles() {
        let err: CrossTalkError = std::fmt::Error.into();
        assert!(matches!(err, CrossTalkError::Fmt(_)));
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn string_and_str_conversions_compile() {
        let from_string: CrossTalkError = String::from("bad config").into();
        let from_str: CrossTalkError = "bad config".into();
        assert!(matches!(from_string, CrossTalkError::Config(_)));
        assert!(matches!(from_str, CrossTalkError::Config(_)));
        assert!(!from_string.to_string().is_empty());
        assert!(!from_str.to_string().is_empty());
    }

    #[test]
    fn boxed_other_conversion_compiles() {
        let boxed: Box<dyn std::error::Error + Send + Sync + 'static> =
            Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "nope"));
        let err: CrossTalkError = boxed.into();
        assert!(matches!(err, CrossTalkError::Other(_)));
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn display_outputs_are_non_empty_for_all_variants() {
        let variants = [
            CrossTalkError::Io(std::io::Error::new(std::io::ErrorKind::Other, "x")),
            CrossTalkError::Fmt(std::fmt::Error),
            CrossTalkError::Config("c".to_string()),
            CrossTalkError::Agent("a".to_string()),
            CrossTalkError::Other(Box::new(std::fmt::Error)),
        ];
        for v in &variants {
            assert!(!v.to_string().is_empty(), "Display must be non-empty");
        }
    }

    #[test]
    fn implements_std_error_trait() {
        fn assert_is_error<E: std::error::Error>() {}
        assert_is_error::<CrossTalkError>();

        // source() returns Some for wrapping variants.
        let err = CrossTalkError::Io(std::io::Error::new(std::io::ErrorKind::Other, "y"));
        assert!(err.source().is_some());

        let err = CrossTalkError::Config("z".to_string());
        assert!(err.source().is_none());
    }

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CrossTalkError>();
    }
}