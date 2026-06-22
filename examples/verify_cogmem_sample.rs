//! Cross-implementation verification of the cogmem C2PA agent-credential sample.
//!
//! The bytes in `examples/fixtures/*.cose` are the exact COSE/SCITT cognition
//! statements embedded in cogmem's public C2PA sample
//! (`cogmem/examples/c2pa-agent-credential/agent-content.c2pa`). This binary
//! re-verifies them with crosstalk's own COSE/SCITT verifier — the same one that
//! checks live orchestration turns — proving the shared substrate is byte-compatible
//! across implementations. The reasoning audit is crosstalk's own statement type;
//! the cogmem memory statement verifies under the identical substrate.
//!
//! Run: `cargo run --example verify_cogmem_sample`

use ciborium::value::Value;
use crosstalk::engines::security::verify_orchestration_audit_statement;

const MEMORY: &[u8] = include_bytes!("fixtures/cogmem.memory.provenance.cose");
const REASONING: &[u8] = include_bytes!("fixtures/crosstalk.orchestration.audit.cose");

fn field<'a>(claim: &'a Value, key: &str) -> Option<&'a Value> {
    match claim {
        Value::Map(entries) => entries
            .iter()
            .find(|(k, _)| matches!(k, Value::Text(t) if t == key))
            .map(|(_, v)| v),
        _ => None,
    }
}

fn text(claim: &Value, key: &str) -> String {
    match field(claim, key) {
        Some(Value::Text(t)) => t.clone(),
        _ => "?".to_string(),
    }
}

fn main() -> anyhow::Result<()> {
    println!(
        "crosstalk independently verifying the cogmem C2PA sample's cognition \
         statements:\n"
    );

    let rsn = verify_orchestration_audit_statement(REASONING)?;
    let turns = match field(&rsn, "turn_count") {
        Some(Value::Integer(i)) => {
            let n: i128 = (*i).into();
            n.to_string()
        }
        _ => "?".to_string(),
    };
    println!("  VERIFIED reasoning crosstalk.orchestration.audit");
    println!("           issuer {}", text(&rsn, "iss"));
    println!(
        "           attests: session '{}', {} turns",
        text(&rsn, "session_id"),
        turns
    );

    let mem = verify_orchestration_audit_statement(MEMORY)?;
    println!("  VERIFIED memory    cogmem.memory.provenance");
    println!("           issuer {}", text(&mem, "iss"));
    println!(
        "           attests: memory '{}' ({}, {})",
        text(&mem, "memoryId"),
        text(&mem, "memoryType"),
        text(&mem, "event")
    );

    println!(
        "\nPASS: both cognition statements verify under crosstalk — identical bytes,\n      \
         independent implementation. Cross-implementation conformance confirmed."
    );
    Ok(())
}
