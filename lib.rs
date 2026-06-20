# File: crosstalk-concurrency/Cargo.toml
[package]
name = "crosstalk-concurrency"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
ratatui = "0.28"
rustc-hash = "2"
moka = { version = "0.12", features = ["future"] }

[dev-dependencies]
tokio = { version = "1", features = ["full", "test-util", "macros", "rt-multi-thread"] }