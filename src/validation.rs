use thiserror::Error;
use tree_sitter::Parser;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Parse error in {language}: line {line}, col {column}: {message}")]
    ParseError {
        language: String,
        line: usize,
        column: usize,
        message: String,
    },
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
}

pub struct AstValidator;

impl AstValidator {
    /// Validates the content using the appropriate tree-sitter grammar.
    /// Currently only supports Rust.
    /// # Errors
    /// Returns `ValidationError` if parsing fails or language is unsupported.
    pub fn validate(content: &str, language: &str) -> Result<(), ValidationError> {
        match language.to_lowercase().as_str() {
            "rust" | "rs" => Self::validate_rust(content),
            _ => Err(ValidationError::UnsupportedLanguage(language.to_string())),
        }
    }

    fn validate_rust(content: &str) -> Result<(), ValidationError> {
        let mut parser = Parser::new();
        let language = tree_sitter_rust::LANGUAGE;
        parser
            .set_language(&language.into())
            .map_err(|e| ValidationError::ParseError {
                language: "rust".to_string(),
                line: 0,
                column: 0,
                message: format!("Failed to load rust grammar: {e:?}"),
            })?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| ValidationError::ParseError {
                language: "rust".to_string(),
                line: 0,
                column: 0,
                message: "Failed to parse content".to_string(),
            })?;

        let root_node = tree.root_node();
        if root_node.has_error() {
            // Find the first error node
            let mut cursor = root_node.walk();
            let mut error_found = false;
            let mut line = 0;
            let mut column = 0;

            loop {
                let node = cursor.node();
                if node.is_error() || node.is_missing() {
                    let start = node.start_position();
                    line = start.row + 1;
                    column = start.column + 1;
                    error_found = true;
                    break;
                }
                if !cursor.goto_first_child() {
                    while !cursor.goto_next_sibling() {
                        if !cursor.goto_parent() {
                            break;
                        }
                    }
                }
                if cursor.node() == root_node {
                    break;
                }
            }

            if error_found {
                return Err(ValidationError::ParseError {
                    language: "rust".to_string(),
                    line,
                    column,
                    message: "Syntax error".to_string(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_rust() {
        let code = "fn main() { println!(\"Hello\"); }";
        assert!(AstValidator::validate(code, "rust").is_ok());
    }

    #[test]
    fn test_validate_invalid_rust() {
        let code = "fn main() { println!(\"Hello\") "; // Missing closing brace
        let result = AstValidator::validate(code, "rust");
        assert!(result.is_err());
        if let Err(ValidationError::ParseError { line, .. }) = result {
            assert!(line > 0);
        } else {
            panic!("Expected ParseError");
        }
    }

    #[test]
    fn test_validate_unsupported_language() {
        let code = "print('hello')";
        let result = AstValidator::validate(code, "python");
        assert!(matches!(
            result,
            Err(ValidationError::UnsupportedLanguage(_))
        ));
    }
}
