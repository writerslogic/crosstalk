use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::Write;

pub struct VerboseLogger {
    file: File,
}

impl VerboseLogger {
    pub fn new(path: &str) -> Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }

    /// ∀ event ⇒ Log(stream, verbose)
    pub fn log(&mut self, source: &str, message: &str) -> Result<()> {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        writeln!(self.file, "[{}] {}: {}", timestamp, source, message)?;
        self.file.flush()?; // Ensure real-time tailing works
        Ok(())
    }
}
