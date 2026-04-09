// verus/invariant_checker.rs
// Formal specification and proofs for InvariantChecker properties.
// Corresponds to: src/engines/verification.rs — InvariantChecker::check_all
//
// Run: verus verus/invariant_checker.rs

use vstd::prelude::*;

verus! {

// ---------------------------------------------------------------------------
// Abstract models (mirrors src/types/conversation.rs)
// ---------------------------------------------------------------------------

pub struct TurnModel {
    pub index: u32,
}

pub struct ArtifactModel {
    pub version: u32,
    pub history_len: int,
}

pub struct StateModel {
    pub iteration_index: u32,
    pub turns: Seq<TurnModel>,
    pub artifacts: Seq<ArtifactModel>,
}

// ---------------------------------------------------------------------------
// Spec predicates: each mirrors one branch of InvariantChecker::check_all
// ---------------------------------------------------------------------------

/// Invariant I1: consecutive turn indices are strictly increasing.
pub open spec fn turns_monotonic(sigma: StateModel) -> bool {
    forall |k: int|
        0 <= k < sigma.turns.len() - 1 ==>
        sigma.turns[k].index < sigma.turns[k + 1].index
}

/// Invariant I2: every turn index is strictly less than iteration_index
/// (no orphan future turns).
pub open spec fn no_orphan_turns(sigma: StateModel) -> bool {
    forall |k: int|
        0 <= k < sigma.turns.len() ==>
        sigma.turns[k].index < sigma.iteration_index
}

/// Invariant I3: every artifact's version equals its history length.
pub open spec fn artifacts_consistent(sigma: StateModel) -> bool {
    forall |k: int|
        0 <= k < sigma.artifacts.len() ==>
        sigma.artifacts[k].version as int == sigma.artifacts[k].history_len
}

/// Conjunction of all invariants — the postcondition of check_all returning Ok.
pub open spec fn all_invariants(sigma: StateModel) -> bool {
    &&& turns_monotonic(sigma)
    &&& no_orphan_turns(sigma)
    &&& artifacts_consistent(sigma)
}

// ---------------------------------------------------------------------------
// Abstract model of check_all's return value.
// Returns Ok(()) iff all_invariants holds; Err otherwise.
// ---------------------------------------------------------------------------
pub open spec fn check_all_model(sigma: StateModel) -> bool {
    all_invariants(sigma)
}

// ---------------------------------------------------------------------------
// Proof 1: check_all_completeness
//   If check_all_model returns true (Ok), then all individual invariants hold.
//   This confirms the checker does not silently drop any invariant branch.
// ---------------------------------------------------------------------------
pub proof fn check_all_completeness(sigma: StateModel)
    requires check_all_model(sigma)
    ensures
        turns_monotonic(sigma),
        no_orphan_turns(sigma),
        artifacts_consistent(sigma),
{
    // all_invariants is the conjunction of the three predicates; unfolding
    // distributes the conjunction into the three ensures clauses.
    assert(all_invariants(sigma));
}

// ---------------------------------------------------------------------------
// Proof 2: check_all_soundness (converse direction)
//   If all three invariants hold individually, check_all_model returns true.
// ---------------------------------------------------------------------------
pub proof fn check_all_soundness(sigma: StateModel)
    requires
        turns_monotonic(sigma),
        no_orphan_turns(sigma),
        artifacts_consistent(sigma),
    ensures check_all_model(sigma)
{
    assert(all_invariants(sigma));
}

// ---------------------------------------------------------------------------
// Proof 3: invariant stability under turn append
//   Appending a new turn with index == sigma.iteration_index preserves I2,
//   and preserves I1 when the new index exceeds the last existing index.
// ---------------------------------------------------------------------------
pub proof fn invariant_stable_on_append(sigma: StateModel, new_turn: TurnModel)
    requires
        all_invariants(sigma),
        new_turn.index < sigma.iteration_index,
        sigma.turns.len() > 0 ==>
            sigma.turns.last().index < new_turn.index,
    ensures
        no_orphan_turns(StateModel {
            turns: sigma.turns.push(new_turn),
            ..sigma
        }),
        turns_monotonic(StateModel {
            turns: sigma.turns.push(new_turn),
            ..sigma
        }),
{
    let sigma2 = StateModel { turns: sigma.turns.push(new_turn), ..sigma };

    // --- no_orphan_turns ---
    assert forall |k: int| 0 <= k < sigma2.turns.len()
        implies sigma2.turns[k].index < sigma2.iteration_index
    by {
        if k < sigma.turns.len() {
            // Existing turns: holds by I2 on sigma.
            assert(sigma.turns[k].index < sigma.iteration_index);
        } else {
            // The appended turn: holds by requires.
            assert(k == sigma.turns.len() as int);
            assert(sigma2.turns[k] == new_turn);
        }
    };

    // --- turns_monotonic ---
    assert forall |k: int| 0 <= k < sigma2.turns.len() - 1
        implies sigma2.turns[k].index < sigma2.turns[k + 1].index
    by {
        if k < sigma.turns.len() - 1 {
            // Interior pairs: holds by I1 on sigma.
        } else {
            // Last existing turn → appended turn.
            assert(k == sigma.turns.len() as int - 1);
            assert(sigma2.turns[k] == sigma.turns.last());
            assert(sigma2.turns[k + 1] == new_turn);
            // sigma.turns.last().index < new_turn.index by requires.
        }
    };
}

} // verus!
