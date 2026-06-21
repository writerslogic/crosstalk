# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Tamper-evident session transcripts: ed25519-signed turns with a persisted,
  optionally passphrase-encrypted (ChaCha20-Poly1305 + Argon2id) key.
- Out-of-band public-key pinning (`CROSSTALK_EXPECTED_PUBKEY`) and key-only
  verification via `TurnVerifier`.
- Keyless transcript hash chain anchored to git commit messages, verified on resume.
- Sandbox resource-limit reporting (fuel consumed, elapsed, resource-limit kills).
- Signed, risk-classified audit log and prompt-injection screening on the tool path.

### Changed
- Split the orchestrator into a directory module; consolidated cross-session
  snapshot persistence.

### Removed
- Dead MCP bridge API and unused concurrency primitives.
