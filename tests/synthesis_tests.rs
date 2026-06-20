use crosstalk::engines::diff::DiffEngine;
use crosstalk::engines::reasoning::SynthesisEngine;

#[test]
fn test_synthesis_engine_fails_to_add_new_function() {
    let base = r#"fn main() {
    println!("hello");
}
"#;

    // Agent 1: Adds a new helper function
    let v1 = r#"fn main() {
    println!("hello");
}

fn helper1() {
    println!("helper1");
}
"#;

    // Agent 2: Adds the same new helper function
    let v2 = r#"fn main() {
    println!("hello");
}

fn helper1() {
    println!("helper1");
}
"#;

    // Agent 3: Adds a DIFFERENT helper function
    let v3 = r#"fn main() {
    println!("hello");
}

fn helper2() {
    println!("helper2");
}
"#;

    let diff1 = DiffEngine::generate_delta(base, v1, 0);
    let diff2 = DiffEngine::generate_delta(base, v2, 0);
    let diff3 = DiffEngine::generate_delta(base, v3, 0);

    let merged = SynthesisEngine::merge(
        base,
        vec![
            ("agent1".to_string(), diff1),
            ("agent2".to_string(), diff2),
            ("agent3".to_string(), diff3),
        ],
        "rust",
    )
    .expect("Should merge successfully");

    // EXPECTATION: Since 2 out of 3 agents agreed on helper1, it SHOULD be in the output.
    // BUG: Current implementation only iterates over base_blocks, so it will MISS helper1.
    assert!(
        merged.contains("fn helper1()"),
        "Merged output should contain helper1 but got:
{}",
        merged
    );
}

#[test]
fn test_synthesis_engine_deletion_consensus() {
    let base = r#"fn main() {}

fn old_unused() {}
"#;

    // Agent 1: Deletes old_unused
    let v1 = "fn main() {}\n";

    // Agent 2: Deletes old_unused
    let v2 = "fn main() {}\n";

    // Agent 3: Keeps old_unused
    let v3 = r#"fn main() {}

fn old_unused() {}
"#;

    let diff1 = DiffEngine::generate_delta(base, v1, 0);
    let diff2 = DiffEngine::generate_delta(base, v2, 0);
    let diff3 = DiffEngine::generate_delta(base, v3, 0);

    let merged = SynthesisEngine::merge(
        base,
        vec![
            ("agent1".to_string(), diff1),
            ("agent2".to_string(), diff2),
            ("agent3".to_string(), diff3),
        ],
        "rust",
    )
    .expect("Should merge successfully");

    // EXPECTATION: Since 2 out of 3 agents deleted it, it SHOULD be gone.
    // BUG: Current implementation will likely keep it because it only processes 'changes' relative to base.
    assert!(
        !merged.contains("fn old_unused()"),
        "Merged output should NOT contain old_unused but got:
{}",
        merged
    );
}
