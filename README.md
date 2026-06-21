<p align="center">
  <strong>Crosstalk</strong><br>
  Multi-model AI orchestrator — fan a task across models, synthesize a verified consensus
</p>

<p align="center">
  <a href="https://orcid.org/0009-0003-1849-2963"><img src="https://img.shields.io/badge/ORCID-0009--0003--1849--2963-green.svg" alt="ORCID"></a>
</p>

<p align="center">
  <a href="https://github.com/writerslogic/crosstalk/actions/workflows/ci.yml"><img src="https://github.com/writerslogic/crosstalk/actions/workflows/ci.yml/badge.svg" alt="Build Status"></a>
  <img src="https://img.shields.io/badge/rust-1.91%2B-orange" alt="Rust">
  <img src="https://img.shields.io/badge/edition-2024-orange" alt="Edition 2024">
  <a href="https://github.com/writerslogic/crosstalk/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

---

## Overview

Crosstalk is a **metacognitive multi-agent orchestrator**. It runs several language
models as a reasoning *swarm*, mediates their proposals through an adaptive debate
topology, scores them along multiple objectives, verifies candidate changes in a
sandbox, and synthesizes a single result — while observing its own process and
improving across sessions.

What makes it more than an ensemble:

- A **metacognitive observer** tracks each model's epistemic reliability (Elo-rated),
  detects reasoning fallacies, and injects adversarial challenges when the swarm gets
  too agreeable.
- The **debate topology adapts per turn** — direct implementation, critique, step-by-step,
  ensemble voting, tree-of-thoughts — and shifts strategy automatically on deadlock or
  quality drops, guided by a bandit over past outcomes.
- Prompts **evolve** (DSPy-inspired tournament selection), and Elo ratings, topology
  scores, agent profiles, and distilled session lessons **persist across runs**.
- Every turn is **signed and hash-chained** (anchored to git) under a **fiduciary
  governance** model with signed persona disclosures and data-retention enforcement.

All of it runs from a terminal UI, optionally writing accepted changes back to your source.

## Install

```sh
# From source
git clone https://github.com/writerslogic/crosstalk
cd crosstalk
cargo build --release        # binary at target/release/crosstalk

# Or install directly
cargo install --git https://github.com/writerslogic/crosstalk
```

Requires **Rust 1.91+** (edition 2024) and an API key for at least one provider.

## Quick Start

```sh
# Mediate a task across two models, writing accepted changes back to ./src
crosstalk --task "Fix the failing parser test" \
          --models <anthropic-model-id> <openai-model-id> \
          --workspace . --files src/ --edit

# Let Crosstalk pick models automatically
crosstalk --task "Refactor the auth module" --auto --workspace .

# Resume a prior session (verifies the restored transcript on load)
crosstalk --resume <session-id>
```

## Usage

| Flag | Description |
|------|-------------|
| `-t, --task <TASK>` | The task to mediate |
| `-m, --models <IDS>...` | Model ids to fan the task across |
| `-A, --auto` | Auto-select the best available models |
| `-i, --iterations <N>` | Max mediation rounds (`0` = until convergence) |
| `-w, --workspace <DIR>` | Workspace root for context and edits |
| `-f, --files <GLOBS>...` | Files/globs to load as context |
| `-e, --edit` | Write accepted changes back to source |
| `--resume <SESSION_ID>` | Resume and verify a prior session |
| `--agent-timeout-secs <N>` | Per-model timeout (default 300) |

Run `crosstalk --help` for the complete list.

## Configuration

Set the API key(s) for the providers you want (a `.env` file is loaded if present; see
[`.env.example`](.env.example)). Pass each model by its provider's own id via `--models`;
an id containing `/` (or prefixed `openrouter:`) is routed through OpenRouter.

| Provider | Env var |
|----------|---------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

## Features

**Mediation & topology**
- Fan a task across multiple models and synthesize one result instead of trusting any single one.
- Adaptive debate topology per turn: direct implementation, debate-and-critique,
  step-by-step, ensemble voting, round-robin, adversarial, tree-of-thoughts.
- Automatic topology shifts on deadlock, quality drop, or agent-count change, selected by
  a UCB1 bandit over historical outcomes (see `crosstalk-eval`).

**Metacognition & self-improvement**
- A metacognitive observer (the swarm's "executive function") that Elo-rates each agent's
  reliability, detects reasoning fallacies, and injects adversarial challenges.
- DSPy-inspired evolutionary prompt optimization via tournament selection.
- Cross-session learning: Elo ratings, topology scores, collective agent profiles, recall
  ranker weights, and distilled session lessons persist between runs.

**Scoring & selection**
- Multi-objective reward (Pareto) combining quality, consistency, novelty, surprise, and
  completion signals — not a single scalar.
- Monte Carlo prediction of whether a candidate change will be accepted.

**Verification & safety**
- WASM sandbox (wasmtime) with CPU-fuel, epoch-deadline, memory, and wall-clock limits,
  distinguishing resource-limit kills from ordinary failures.
- Optional Verus formal verification of safety invariants on critical state transitions
  (see [VERUS.md](VERUS.md)).
- Permissioned tool gateway for model-invoked tools (`read_file`, `write_file`,
  allow-listed `shell_exec`, memory queries), confined to the workspace, with a signed
  audit log and prompt-injection screening.

**Provenance & governance**
- Tamper-evident transcripts: each turn is ed25519-signed (persisted, optionally
  passphrase-encrypted key, pinned public identity) and linked into a hash chain anchored
  in git commit messages. See [SECURITY.md](SECURITY.md).
- A fiduciary/principal model with signed persona disclosures and data-retention
  (minimization) enforcement.

**Memory & interface**
- Embedding-based memory with relevance recall across sessions.
- A ratatui terminal UI showing per-model progress, scoring, and synthesis.

## Architecture

```
src/
  core/        orchestrator (agents, synthesis, artifacts, verification, parsing,
               lifecycle), state machine, model factory, config
  engines/     the capability domains, each owning one concern:
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

Each turn flows through *propose → observe → score → adapt → synthesize → verify →
commit*. State is checkpointed to an embedded store every turn; resuming a session
re-verifies its signatures and hash chain before continuing, and rehydrates the
cross-session learning state (Elo, topology, profiles, lessons).

## Development

```sh
cargo build --release
cargo test --workspace                              # full suite
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Security

Crosstalk signs and hash-chains session transcripts and confines model-invoked tools to
the workspace. To report a vulnerability, see [SECURITY.md](SECURITY.md) — please do not
open a public issue for security reports.

## License

Apache-2.0 &copy; [WritersLogic, Inc.](https://github.com/writerslogic)
