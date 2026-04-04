use thiserror::Error;
use tree_sitter::{Parser, Node};
use std::collections::HashMap;

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
    /// Validates the content and extracts top-level nodes for versioning.
    pub fn validate(content: &str, language: &str) -> Result<(), ValidationError> {
        match language.to_lowercase().as_str() {
            "rust" | "rs" => Self::validate_rust(content),
            _ => Err(ValidationError::UnsupportedLanguage(language.to_string())),
        }
    }

    /// Extracts top-level nodes (fn, struct, impl) with their content.
    pub fn extract_nodes(content: &str, language: &str) -> HashMap<String, String> {
        if !matches!(language.to_lowercase().as_str(), "rust" | "rs") {
            return HashMap::new();
        }

        let mut parser = Parser::new();
        let _ = parser.set_language(&tree_sitter_rust::LANGUAGE.into());
        let tree = match parser.parse(content, None) {
            Some(t) => t,
            None => return HashMap::new(),
        };

        let mut nodes = HashMap::new();
        let root = tree.root_node();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            let kind = child.kind();
            if matches!(kind, "function_item" | "struct_item" | "impl_item" | "enum_item") {
                if let Some(name) = Self::get_node_name(child, content) {
                    let id = format!("{}:{}", kind, name);
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        nodes.insert(id, text.to_string());
                    }
                }
            }
        }
        nodes
    }

    fn get_node_name(node: Node, content: &str) -> Option<String> {
        // Find identifier child
        for i in 0..node.child_count() {
            let child = node.child(i).unwrap();
            if child.kind() == "name" || child.kind() == "identifier" || child.kind() == "type_identifier" {
                return child.utf8_text(content.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
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
