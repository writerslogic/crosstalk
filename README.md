# Crosstalk

Multi-model AI orchestrator. Crosstalk routes a task to several language models in
parallel, scores and synthesizes their proposals into a consensus answer, optionally
verifies and writes the result back to your source files, and presents the whole run
in a terminal UI.

> **Status:** active development. The engine is functional and tested
> (`cargo test --workspace` is green, `cargo clippy` is clean), but the project is
> evolving quickly and APIs/behavior may change between commits.

## What it does

- **Fan-out / consensus** — sends a task to multiple models, then scores proposals and
  synthesizes a single result rather than trusting any one model.
- **Verification** — compiles/lints candidate changes in a WASM sandbox before accepting
  them, with CPU-fuel and wall-clock limits.
- **Memory & recall** — persists session transcripts and recalls relevant prior context.
- **Tamper-evident transcripts** — each turn is ed25519-signed (with a persisted,
  optionally passphrase-encrypted key) and linked into a hash chain whose head is
  anchored in git commit messages, so reordering or editing past turns is detectable.
- **Tool gateway** — a permissioned MCP-style interface for model-invoked tools
  (`read_file`, `write_file`, allow-listed `shell_exec`, memory queries) confined to the
  workspace.
- **Terminal UI** — a ratatui interface showing per-model progress, scoring, and synthesis.

## Requirements

- Rust **1.91+** (edition 2024)
- An API key for at least one provider (see below)

## Build

```sh
cargo build --release
cargo test --workspace      # 588 tests
```

## Configure

Set the API key(s) for the providers you want to use (a `.env` file is loaded if present).
Pass each model by its provider's own model id via `--models`:

| Provider   | Env var               |
|------------|-----------------------|
| Anthropic  | `ANTHROPIC_API_KEY`   |
| OpenAI     | `OPENAI_API_KEY`      |
| DeepSeek   | `DEEPSEEK_API_KEY`    |
| Mistral    | `MISTRAL_API_KEY`     |
| Groq       | `GROQ_API_KEY`        |
| OpenRouter | `OPENROUTER_API_KEY`  |

A model id containing `/` (or prefixed `openrouter:`) is routed through OpenRouter.

Optional:

- `CROSSTALK_SIGNING_PASSPHRASE` — encrypt the transcript signing key at rest.
- `CROSSTALK_EXPECTED_PUBKEY` — pin the signing identity out-of-band; a mismatched key aborts.

## Run

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

Key flags: `--task`, `--models`, `--auto`, `--iterations`, `--workspace`, `--files`,
`--edit`, `--resume`, `--agent-timeout-secs`. Run `crosstalk --help` for the full list.

## Workspace layout

- `src/` — the orchestrator (`core/`, `engines/`, `mcp/`, `types/`, `ui/`)
- `crosstalk-concurrency/` — cancellation primitives
- `crosstalk-eval/` — benchmarking harness for topology selection

## License

[AGPL-3.0-only](LICENSE) © David Condrey / Writers Logic.
