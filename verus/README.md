# Crosstalk — Verus Formal Verification

This directory contains formal specifications and SMT-verified proofs for core
invariants of the Crosstalk runtime. The proofs use
[Verus](https://github.com/verus-lang/verus), a Rust-based verification tool
backed by the Z3 SMT solver.

## File map

| File | Rust source | What is proved |
|------|-------------|----------------|
| `state.rs` | `src/core/state.rs` | checkpoint/restore identity, monotonic turns, artifact version consistency |
| `hash_chain.rs` | `src/engines/verification.rs` | hash determinism, chain integrity, verify soundness |
| `invariant_checker.rs` | `src/engines/verification.rs` | check_all completeness and soundness, invariant stability on append |

## Installation

```sh
# Install the Verus toolchain (requires Rust nightly).
git clone https://github.com/verus-lang/verus
cd verus/source
./tools/get-z3.sh        # downloads the Z3 binary for your platform
vargo build --release
# The compiled binary is verus/source/target-verus/release/verus
export PATH="$PATH:$(pwd)/target-verus/release"
```

Verus requires no changes to `Cargo.toml`; the proof files are stand-alone.

## Running the proofs

```sh
# Verify all three files.
verus verus/state.rs
verus verus/hash_chain.rs
verus verus/invariant_checker.rs

# Or verify the whole directory at once.
for f in verus/*.rs; do verus "$f"; done
```

A passing run prints `verification results:: N verified, 0 errors` for each file.

## Proof summaries

### state.rs

**`checkpoint_then_restore_identity`**
Models the DB as `Map<u32, StateModel>`. After inserting `sigma` at
`sigma.iteration_index`, a lookup at that key returns `sigma`. Follows
directly from `vstd`'s `Map::insert` axiom.

**`iteration_monotonic`**
Given a state whose consecutive turns are strictly ordered by index, proves
that all pairs `(i, j)` with `i < j` satisfy `turns[i].index < turns[j].index`.
Uses a recursive helper (`monotonic_transitive`) with a `decreases j - i`
termination argument; Z3 closes each induction step automatically.

**`artifact_version_consistency`**
Documents the invariant that `artifact.version == artifact.history.len()` as a
precondition-to-postcondition pass-through, confirming that
`InvariantChecker::check_all`'s artifact branch faithfully encodes it.

### hash_chain.rs

**`hash_deterministic`**
Pure spec functions are referentially transparent; equal arguments produce equal
results. Verified in zero proof steps.

**`hash_chain_integrity`**
SHA-256 collision resistance cannot be derived from first principles by an SMT
solver. The property is axiomatised via `hash_collision_free` (marked
`#[verifier::external_body]`) and then used as a one-step corollary. This is
standard practice for cryptographic assumptions in formal verification.

**`verify_soundness`**
`HashChain::verify` returns `true` only when the stored hash equals the
recomputed hash. Proved by unfolding `verify_model`.

### invariant_checker.rs

**`check_all_completeness`**
`check_all_model(sigma) == true` implies each of the three individual
predicates (`turns_monotonic`, `no_orphan_turns`, `artifacts_consistent`).
Proved by unfolding the `all_invariants` conjunction.

**`check_all_soundness`**
Converse: all three predicates individually imply `check_all_model`. Together
with completeness this establishes that the model is an exact specification of
the implementation.

**`invariant_stable_on_append`**
Appending a turn whose index is (a) less than `iteration_index` and (b) greater
than the current last turn preserves both `no_orphan_turns` and
`turns_monotonic`. The proof splits on whether the index `k` refers to an
existing turn or the newly appended one.

## Notes

- The `.rs` extension is required by Verus; the spec refers to these files as
  `.v` only informally.
- The proof files contain no `exec` code and are never compiled into the
  Crosstalk binary.
- Cross-references in `src/engines/verification.rs` point back to this
  directory for each verified struct.
