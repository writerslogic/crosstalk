# Security Policy

## Supported Versions

Crosstalk is pre-1.0 and under active development. Security fixes are applied to
`main`; there is no long-term support branch yet.

| Version | Supported |
|---------|-----------|
| `main`  | ✅        |
| < 0.1   | ❌        |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### Preferred Method

Report vulnerabilities via email to: **admin@writerslogic.com**

Include a description, reproduction steps, affected versions/commit, and any
proof-of-concept. Please allow time for a fix before public disclosure.

### Response Timeline

- Acknowledgement within a few business days.
- An assessment and remediation plan once reproduced.

### Disclosure Policy

We follow coordinated disclosure:

1. You report privately.
2. We confirm and develop a fix.
3. We release the fix with credit to the reporter (unless anonymity is requested).
4. Public disclosure after users have had time to update.

## Security Model

Crosstalk runs untrusted model output and confines model-invoked actions. The
relevant guarantees:

### Tool directives

Model-invoked tools (`read_file`, `write_file`, `shell_exec`, memory queries) run
through a permissioned gateway:

- All file paths are confined to the workspace; absolute paths, `~`, and `..`
  traversal are rejected, and writes are re-checked against the canonical root.
- `shell_exec` is restricted to an allow-list of read-only commands and rejects
  shell metacharacters and control characters.
- Untrusted file contents read back into model context are screened for known
  prompt-injection patterns.
- Every tool-directive attempt is recorded in a signed, risk-classified audit log.

### Transcript integrity

- Each turn (and persona disclosure) is **ed25519-signed**. The signing seed is
  persisted and, when `CROSSTALK_SIGNING_PASSPHRASE` is set, encrypted at rest
  with ChaCha20-Poly1305 under an Argon2id-derived key.
- The public key is **pinned**: a seed that does not match the recorded identity
  aborts on load. `CROSSTALK_EXPECTED_PUBKEY` provides an authoritative
  out-of-band pin. Verification uses the public key, never the secret.
- Turns are linked into a **hash chain** whose head is anchored in git commit
  messages, so reorder/insert/delete/edit of past turns is detectable without any
  secret. Both checks run when a session is resumed.

### Sandbox

Candidate changes are executed in a WASM sandbox (wasmtime) with a CPU-fuel
budget, an epoch-deadline interrupt, a memory cap, and a wall-clock timeout;
resource-limit kills are distinguished from ordinary failures.

### Cryptographic Primitives

- Signing: ed25519 (`ed25519-dalek`)
- Seed encryption: ChaCha20-Poly1305 (`chacha20poly1305`)
- Key derivation: Argon2id (`argon2`)
- Hashing: SHA-256 (`sha2`)

Key material is held in `Zeroizing` buffers and never logged.

## Hardening Recommendations

- Set `CROSSTALK_SIGNING_PASSPHRASE` to encrypt the signing key at rest.
- Record the signing identity (logged at startup) and pin it via
  `CROSSTALK_EXPECTED_PUBKEY` in your environment or CI.
- Run Crosstalk against a workspace that is a dedicated git checkout so the
  transcript hash-chain anchor lands in a history you control.
