# Verus Formal Verification

Crosstalk uses [Verus](https://github.com/verus-lang/verus) to prove safety invariants on
critical state-machine transitions.  Verus requires a specific nightly Rust toolchain and
the Z3 SMT solver, so it is **not** run in the standard CI pipeline (see
`.github/workflows/ci.yml` for the explanation).

## Proof files

| File | Invariants proven |
|------|-------------------|
| `proofs/state.rs` | Checkpoint/restore round-trip identity; iteration index is monotonically non-decreasing |
| `proofs/hash_chain.rs` | Append-only chain integrity; SHA-256 collision resistance (axiomatized) |
| `proofs/invariant_checker.rs` | Artifact version consistency; state validity on append |

## Running the proofs locally

### 1. Install Verus

```sh
git clone https://github.com/verus-lang/verus
cd verus
# Follow the "Building Verus" instructions in the Verus README.
# Requires: rustup, z3 >= 4.12 on PATH.
```

### 2. Install Z3

```sh
# macOS
brew install z3

# Ubuntu/Debian
apt install z3

# Windows — download from https://github.com/Z3Prover/z3/releases
```

### 3. Verify the proofs

From the repository root:

```sh
verus proofs/state.rs
verus proofs/hash_chain.rs
verus proofs/invariant_checker.rs
```

A successful run prints `verification results:: N verified, 0 errors` for each file.

## Why proofs are excluded from CI

Verus requires a pinned nightly toolchain that diverges from the stable toolchain used for
the main build.  Installing Verus and Z3 on every CI runner adds several minutes to each
run and would complicate the matrix (macOS, Linux, Windows).  The proofs are stable — they
are re-run manually before every release and whenever the proved functions change.  The CI
pipeline enforces the same safety properties at runtime via the `InvariantChecker` and
`ContinuousAuditor` engines.
