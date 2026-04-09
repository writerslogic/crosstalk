use anyhow::{Result, anyhow};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const CACHE_TTL_SECS: i64 = 3600;

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    created_at: i64,
    env: HashMap<String, String>,
}

pub struct NixManager {
    pub deps: Vec<String>,
    pub cache_dir: PathBuf,
    pub ttl_secs: i64,
}

impl NixManager {
    pub fn new(deps: Vec<String>) -> Self {
        Self {
            deps,
            cache_dir: PathBuf::from("/tmp/.crosstalk-nix-cache"),
            ttl_secs: CACHE_TTL_SECS,
        }
    }

    pub fn generate_flake(&self) -> Result<String> {
        for dep in &self.deps {
            if !dep.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
                return Err(anyhow!("invalid dependency name: {}", dep));
            }
        }
        let deps_str = self
            .deps
            .iter()
            .map(|d| format!("pkgs.{}", d))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(format!(
            r#"{{
  description = "Crosstalk dynamic environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = {{ self, nixpkgs }}: let
    pkgs = nixpkgs.legacyPackages.${{builtins.currentSystem}};
  in {{
    devShells.${{builtins.currentSystem}}.default = pkgs.mkShell {{
      buildInputs = [ {deps} ];
    }};
  }};
}}"#,
            deps = deps_str
        ))
    }

    pub fn synthesize(&self) -> Result<HashMap<String, String>> {
        let cache_path = self.cache_path();

        if let Some(cached) = self.load_cache(&cache_path) {
            return Ok(cached);
        }

        let flake_content = self.generate_flake()?;
        let tmp_dir = std::env::temp_dir().join(format!("crosstalk-nix-{}", self.cache_key()));
        fs::create_dir_all(&tmp_dir)?;
        let flake_path = tmp_dir.join("flake.nix");
        fs::write(&flake_path, &flake_content)?;

        let output = Command::new("nix")
            .args([
                "print-dev-env",
                "--json",
                "--no-warn-dirty",
                tmp_dir.to_str().unwrap_or("."),
            ])
            .output()
            .map_err(|e| anyhow!("Failed to run nix print-dev-env: {}", e))?;

        let _ = fs::remove_dir_all(&tmp_dir);

        if !output.status.success() {
            return Err(anyhow!(
                "nix print-dev-env failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let mut env = HashMap::new();

        if let Some(vars) = json["variables"].as_object() {
            for (k, v) in vars {
                if let Some(val) = v["value"].as_str() {
                    env.insert(k.clone(), val.to_string());
                }
            }
        }

        self.write_cache(&cache_path, &env)?;
        Ok(env)
    }

    pub fn get_env(&self, key: &str) -> Option<String> {
        self.synthesize().ok()?.remove(key)
    }

    fn cache_key(&self) -> String {
        let mut sorted = self.deps.clone();
        sorted.sort();
        let input = sorted.join(",");
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn cache_path(&self) -> PathBuf {
        self.cache_dir.join(format!("{}.json", self.cache_key()))
    }

    fn load_cache(&self, path: &PathBuf) -> Option<HashMap<String, String>> {
        let data = fs::read(path).ok()?;
        let entry: CacheEntry = serde_json::from_slice(&data).ok()?;
        let age = Utc::now().timestamp() - entry.created_at;
        if age >= 0 && age < self.ttl_secs {
            Some(entry.env)
        } else {
            None
        }
    }

    fn write_cache(&self, path: &PathBuf, env: &HashMap<String, String>) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let entry = CacheEntry {
            created_at: Utc::now().timestamp(),
            env: env.clone(),
        };
        let data = serde_json::to_vec(&entry)?;
        fs::write(path, data)?;
        Ok(())
    }
}
