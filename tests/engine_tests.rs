use crosstalk::engines::security::{SecretScanner, ShellSanity, TurnSigner};
use crosstalk::engines::validation::AstValidator;

#[test]
fn test_generate_skeleton() {
    let code = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
        struct Point { x: i32, y: i32 }
        impl Point {
            fn new() -> Self { Point { x: 0, y: 0 } }
        }
    "#;
    let skeleton = AstValidator::generate_skeleton(code, "rust");
    assert!(skeleton.contains("pub fn add(a: i32, b: i32) -> i32 { ... }"));
    assert!(skeleton.contains("struct Point { x: i32, y: i32 }"));
    assert!(skeleton.contains("impl Point {"));
    assert!(skeleton.contains("fn new() -> Self { ... }"));
    assert!(!skeleton.contains("a + b"));
}

#[test]
fn test_secret_scanner() {
    let content = "My key is AKIA1234567890ABCDEF";
    assert_eq!(SecretScanner::scan(content).len(), 1);
}

#[test]
fn test_shell_sanity() {
    assert!(ShellSanity::is_dangerous("rm -rf /"));
    assert!(!ShellSanity::is_dangerous("cargo test"));
}

#[test]
fn test_turn_signer() {
    let signer = TurnSigner::new();
    let data = b"turn data";
    let sig = signer.sign(data);
    assert!(signer.verify(data, &sig));
}
