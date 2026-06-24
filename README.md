<p align="center">
  <img src="https://raw.githubusercontent.com/writerslogic/crosstalk/main/assets/logo-spin.gif" width="200" alt="Crosstalk">
</p>

# Crosstalk

[![CI](https://github.com/writerslogic/crosstalk/actions/workflows/ci.yml/badge.svg)](https://github.com/writerslogic/crosstalk/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/writerslogic/crosstalk/blob/main/LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--1849--2963-green.svg)](https://orcid.org/0009-0003-1849-2963)

**Multi-model AI orchestrator — fan a task across models, synthesize a verified consensus.**

Crosstalk is a metacognitive multi-model orchestrator. It runs several language models as a reasoning swarm, mediates their proposals through an adaptive debate topology, scores them along multiple objectives, verifies candidate changes in a sandbox, and synthesizes a single result — while observing its own process and improving across sessions. Every turn's reasoning is cryptographically signed and hash-chained, so the orchestration audit can be verified, resumed, and bound to the output it produced.

> Developed by [WritersLogic](https://github.com/writerslogic)

## Installation

```bash
# Build from source
git clone https://github.com/writerslogic/crosstalk
cd crosstalk
cargo build --release   # binary at target/release/crosstalk

# Install directly
cargo install --git https://github.com/writerslogic/crosstalk
```

Requires Rust 1.91+ (edition 2024) and an API key for at least one provider.

## Quick Start

```bash
# Mediate a task across two models, write accepted changes back to ./src
crosstalk --task "Fix the failing parser test" \
          --models claude-sonnet-4-6 gpt-4o \
          --workspace . --files src/ --edit

# Let Crosstalk pick models automatically
crosstalk --task "Refactor the auth module" --auto --workspace .

# Resume a prior session (verifies the restored transcript on load)
crosstalk --resume <session-id>
```

## Configuration

Set API keys for the providers you want (`.env` is loaded if present; see [`.env.example`](.env.example)):

| Provider | Env var |
|----------|---------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

Pass model IDs via `--models`. An ID containing `/` or prefixed `openrouter:` routes through OpenRouter.

## CLI Reference

| Flag | Description |
|------|-------------|
| `-t, --task <TASK>` | The task to mediate |
| `-m, --models <IDS>...` | Model IDs to fan the task across |
| `-A, --auto` | Auto-select the best available models |
| `-i, --iterations <N>` | Max mediation rounds (`0` = until convergence) |
| `-w, --workspace <DIR>` | Workspace root for context and edits |
| `-f, --files <GLOBS>...` | Files/globs to load as context |
| `-e, --edit` | Write accepted changes back to source |
| `--resume <SESSION_ID>` | Resume and verify a prior session |
| `--agent-timeout-secs <N>` | Per-model timeout (default 300) |

Run `crosstalk --help` for the complete list.

## Why Crosstalk?

Single-model AI tools give you one perspective. Crosstalk treats that as a failure mode. Instead of trusting any single model, Crosstalk:

- Fans a task across multiple models simultaneously
- Mediates their proposals through an adaptive debate topology
- Scores candidates on multiple objectives (quality, consistency, novelty, surprise, completion)
- Verifies candidates in a WASM sandbox before accepting them
- Synthesizes one result from the best ideas across all models
- Signs and hash-chains every turn so the reasoning is auditable

## Features

### Mediation and Topology

- Fan a task across multiple models and synthesize one result.
- Adaptive debate topology per turn: direct implementation, debate-and-critique, step-by-step, ensemble voting, round-robin, adversarial, tree-of-thoughts.
- Automatic topology shifts on deadlock, quality drop, or agent-count change, selected by a UCB1 bandit over historical outcomes (see `crosstalk-eval`).

### Metacognition and Self-Improvement

- A metacognitive observer (the swarm's executive function) that Elo-rates each agent's reliability, detects reasoning fallacies, and injects adversarial challenges when the swarm converges too quickly.
- DSPy-inspired evolutionary prompt optimization via tournament selection.
- Cross-session learning: Elo ratings, topology scores, collective agent profiles, recall ranker weights, and distilled session lessons persist between runs.

### Scoring and Selection

- Multi-objective reward (Pareto) combining quality, consistency, novelty, surprise, and completion signals — not a single scalar.
- Monte Carlo prediction of whether a candidate change will be accepted.

### Verification and Safety

- WASM sandbox (wasmtime) with CPU-fuel, epoch-deadline, memory, and wall-clock limits, distinguishing resource-limit kills from ordinary failures.
- Optional Verus formal verification of safety invariants on critical state transitions (see [VERUS.md](VERUS.md)).
- Permissioned tool gateway for model-invoked tools (`read_file`, `write_file`, allow-listed `shell_exec`, memory queries), confined to the workspace, with a signed audit log and prompt-injection screening.

### Provenance and Governance

- Tamper-evident transcripts: each turn is Ed25519-signed and linked into a hash chain anchored in git commit messages.
- A portable COSE/SCITT orchestration-audit statement: each session's hash-chain head is emitted as an untagged `COSE_Sign1` (EdDSA, CBOR claim) on the shared provenance substrate — byte-compatible with cogmem and holographic-memory by construction. External verifiers can confirm the reasoning that produced an output without the session store.
- A fiduciary/principal model with signed persona disclosures and data-retention (minimization) enforcement.

### Memory and Interface

- Embedding-based memory with relevance recall across sessions.
- ratatui terminal UI showing per-model progress, scoring, and synthesis.

## Provenance — the Orchestration-Audit Statement

Crosstalk treats its own reasoning as provenance. Each session emits its hash-chain head as a single, portable signed statement that any party can verify independently.

What is implemented (real crypto, tested):

- **Ed25519 signing identity**, exposed as a `did:key` byte-identical to cogmem's (`src/engines/security.rs`).
- **COSE/SCITT orchestration-audit statement**: `orchestration_audit_statement` emits an untagged `COSE_Sign1` — EdDSA (alg -8), content type `application/cbor`, `kid` = the raw 32-byte verifying key, empty external AAD — over a CBOR claim committing the audit root, session ID, turn count, and timestamp. Byte-compatible with cogmem and holographic-memory by construction.
- **Anchoring**: the statement commits the same hash-chain head that the transcript anchors in git, so the portable audit and the resumable session agree.

**Verify it yourself:**

```bash
cargo run --example verify_cogmem_sample
```

Re-verifies the exact COSE/SCITT cognition statements from cogmem's public C2PA sample with crosstalk's own verifier — identical bytes, independent implementation.

## Architecture

```
src/
  core/        orchestrator (agents, synthesis, artifacts, verification, lifecycle)
  engines/     capability domains:
                 metacognition  collective_intelligence  swarm  topology
                 reasoning      consensus  intelligence (reward)  novelty  surprise
                 prompt_evolution  self_improvement  simulation  planning
                 quality  memory  sandbox  security  verification  proof
                 data_minimizer  compute  analytics  release  diff  linter
  mcp/         permissioned tool gateway + CLI bridge
  types/       conversation, artifact, fiduciary, intelligence, mcp, ...
  ui/          ratatui TUI
crosstalk-concurrency/   cancellation primitives
crosstalk-eval/          UCB1 topology-selection benchmarking harness
```

Each turn flows through: propose -> observe -> score -> adapt -> synthesize -> verify -> commit.

State is checkpointed every turn; resuming a session re-verifies its signatures and hash chain before continuing and rehydrates cross-session learning state (Elo, topology, profiles, lessons).

## Part of the Agent-Provenance Stack

Crosstalk is one component of the WritersLogic verifiable agent-provenance pipeline — agent identity, memory, reasoning, and signed output, cryptographically bound end to end.

| Project | Role |
|---|---|
| [cogmem](https://github.com/writerslogic/cogmem) | Agent identity (CAWG credential) + verifiable, tamper-evident memory (COSE/SCITT) |
| **crosstalk (this repo)** | Multi-model orchestrator; signs each turn's reasoning/orchestration audit |
| [holographic-memory](https://github.com/writerslogic/holographic-memory) | Durable holographic memory store; cross-verifies signed statements and agent identity |
| WritersProof | C2PA producer: binds identity + memory + reasoning to the signed asset |

All four share one substrate — COSE_Sign1 / SCITT (Ed25519) and W3C DID identity — specified in [UNIFIED-PROVENANCE.md](https://github.com/writerslogic/cogmem/blob/main/UNIFIED-PROVENANCE.md).

## Development

```bash
cargo build --release
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Security

Crosstalk signs and hash-chains session transcripts and confines model-invoked tools to the workspace. To report a vulnerability, see [SECURITY.md](SECURITY.md) — do not open a public issue for security reports.

## License

Apache-2.0 — see [LICENSE](LICENSE).
