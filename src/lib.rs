pub mod core;
pub mod engines;
pub mod mcp;
pub mod types;
pub mod ui;

#[macro_export]
macro_rules! log_warn {
    ($expr:expr, $msg:expr) => {
        if let Err(e) = $expr {
            tracing::warn!("{}: {:?}", $msg, e);
        }
    };
}
