use serde::Deserialize;

#[derive(Debug, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub default_models: Option<Vec<String>>,
    #[serde(default)]
    pub default_workspace: Option<String>,
    #[serde(default)]
    pub default_iterations: Option<u32>,
    #[serde(default)]
    pub agent_timeout_secs: Option<u64>,
    #[serde(default)]
    pub auto_mode: Option<bool>,
}

impl Config {
    pub fn load() -> Self {
        let config_dir = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{h}/.config"))
                .unwrap_or_else(|_| ".config".to_string())
        });
        let path = format!("{config_dir}/crosstalk/config.toml");
        match std::fs::read_to_string(&path) {
            Ok(content) => toml::from_str(&content).unwrap_or_else(|e| {
                tracing::warn!(path = %path, error = %e, "invalid config file, using defaults");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }
}
