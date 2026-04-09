// verus/hash_chain.rs
// Formal specification and proofs for HashChain correctness.
// Corresponds to: src/engines/verification.rs — HashChain
//
// Run: verus verus/hash_chain.rs

use vstd::prelude::*;

verus! {

// ---------------------------------------------------------------------------
// Abstract model
// ---------------------------------------------------------------------------

/// Opaque representation of a serialized ConversationState.
/// Two states are equal iff their serializations are equal (Bincode is
/// deterministic over BTreeMap / Vec, as required by the production code).
pub type StateBytes = Seq<u8>;

/// A 32-byte hash value.
pub type HashBytes = Seq<u8>;

/// Abstract hash function: SHA-256(serialize(state) ++ prev_hash).
/// Modelled as a pure, uninterpreted spec function.
pub uninterp spec fn hash_model(state: StateBytes, prev: HashBytes) -> HashBytes;

// ---------------------------------------------------------------------------
// Axiom: the hash function is collision-free (models SHA-256 pre-image
// resistance).  Marked external_body because SMT cannot derive cryptographic
// properties; the axiom captures the security assumption.
// ---------------------------------------------------------------------------
#[verifier::external_body]
pub proof fn hash_collision_free(
    s1: StateBytes, s2: StateBytes, prev: HashBytes,
)
    requires s1 != s2
    ensures hash_model(s1, prev) != hash_model(s2, prev)
{}

// ---------------------------------------------------------------------------
// Proof 1: hash_deterministic
//   HashChain::compute is a pure function: identical inputs yield identical
//   outputs.  This follows from referential transparency of spec functions.
// ---------------------------------------------------------------------------
pub proof fn hash_deterministic(
    s1: StateBytes, s2: StateBytes, h1: HashBytes, h2: HashBytes,
)
    requires s1 == s2, h1 == h2
    ensures hash_model(s1, h1) == hash_model(s2, h2)
{
    // Referential transparency: equal arguments produce equal results.
}

// ---------------------------------------------------------------------------
// Proof 2: hash_chain_integrity
//   If the state serialization changes, the resulting hash changes.
// ---------------------------------------------------------------------------
pub proof fn hash_chain_integrity(
    s1: StateBytes, s2: StateBytes, prev: HashBytes,
)
    requires s1 != s2
    ensures hash_model(s1, prev) != hash_model(s2, prev)
{
    hash_collision_free(s1, s2, prev);
}

// ---------------------------------------------------------------------------
// Proof 3: verify_soundness
//   HashChain::verify returns true only when the stored hash matches the
//   recomputed hash for the same (state, prev_hash) pair.
// ---------------------------------------------------------------------------
pub open spec fn verify_model(
    state: StateBytes, prev: HashBytes, stored: HashBytes,
) -> bool {
    hash_model(state, prev) == stored
}

pub proof fn verify_soundness(state: StateBytes, prev: HashBytes, stored: HashBytes)
    requires verify_model(state, prev, stored)
    ensures hash_model(state, prev) == stored
{
    // Unfolds directly from the spec function definition.
}

} // verus!
