use std::collections::HashMap;
use thiserror::Error;
use tree_sitter::{Node, Parser};

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
            if matches!(
                kind,
                "function_item" | "struct_item" | "impl_item" | "enum_item"
            ) && let Some(name) = Self::get_node_name(child, content)
            {
                let id = format!("{}:{}", kind, name);
                if let Ok(text) = child.utf8_text(content.as_bytes()) {
                    nodes.insert(id, text.to_string());
                }
            }
        }
        nodes
    }

    /// Generates a semantic skeleton (signatures only) for the given content.
    pub fn generate_skeleton(content: &str, language: &str) -> String {
        if !matches!(language.to_lowercase().as_str(), "rust" | "rs") {
            return content.lines().take(10).collect::<Vec<_>>().join("\n") + "\n...";
        }

        let mut parser = Parser::new();
        let _ = parser.set_language(&tree_sitter_rust::LANGUAGE.into());
        let tree = match parser.parse(content, None) {
            Some(t) => t,
            None => return String::new(),
        };

        let mut skeleton = String::new();
        let root = tree.root_node();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "function_item" => {
                    // Extract signature (everything up to the block)
                    if let Some(body) = child.child_by_field_name("body") {
                        let start = child.start_byte();
                        let end = body.start_byte();
                        let sig = &content[start..end];
                        skeleton.push_str(sig.trim_end());
                        skeleton.push_str(" { ... }\n");
                    }
                }
                "struct_item" | "enum_item" | "trait_item" => {
                    // For now, include the full definition as they are usually small
                    if let Ok(text) = child.utf8_text(content.as_bytes()) {
                        skeleton.push_str(text);
                        skeleton.push('\n');
                    }
                }
                "impl_item" => {
                    // Extract impl header and method signatures
                    let mut impl_text = String::new();
                    if let Some(body) = child.child_by_field_name("body") {
                        let start = child.start_byte();
                        let end = body.start_byte();
                        let header = &content[start..end];
                        impl_text.push_str(header.trim_end());
                        impl_text.push_str(" {\n");

                        // Extract signatures of methods inside the impl
                        let mut inner_cursor = body.walk();
                        for method in body.children(&mut inner_cursor) {
                            if method.kind() == "function_item"
                                && let Some(m_body) = method.child_by_field_name("body")
                            {
                                let m_start = method.start_byte();
                                let m_end = m_body.start_byte();
                                let m_sig = &content[m_start..m_end];
                                impl_text.push_str("    ");
                                impl_text.push_str(m_sig.trim_end());
                                impl_text.push_str(" { ... }\n");
                            }
                        }
                        impl_text.push_str("}\n");
                    }
                    skeleton.push_str(&impl_text);
                }
                _ => {}
            }
        }
        skeleton
    }

    /// Compares two versions of content and returns the IDs of nodes that changed.
    pub fn identify_changed_nodes(
        old_content: &str,
        new_content: &str,
        language: &str,
    ) -> Vec<String> {
        let old_nodes = Self::extract_nodes(old_content, language);
        let new_nodes = Self::extract_nodes(new_content, language);
        let mut changed = vec![];

        for (id, content) in &new_nodes {
            if let Some(old_val) = old_nodes.get(id) {
                if old_val != content {
                    changed.push(id.clone());
                }
            } else {
                changed.push(id.clone()); // New node
            }
        }

        // Also check for deleted nodes
        for id in old_nodes.keys() {
            if !new_nodes.contains_key(id) {
                changed.push(id.clone());
            }
        }

        changed
    }

    fn get_node_name(node: Node, content: &str) -> Option<String> {
        // Find identifier child
        for i in 0..node.child_count() {
            let child = node.child(i).unwrap();
            if child.kind() == "name"
                || child.kind() == "identifier"
                || child.kind() == "type_identifier"
            {
                return child
                    .utf8_text(content.as_bytes())
                    .ok()
                    .map(|s| s.to_string());
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
                    message: "Syntax error in Rust code".to_string(),
                });
            }
        }
        Ok(())
    }
}
