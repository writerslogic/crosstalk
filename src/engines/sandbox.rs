use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Instant;
use wasmtime::*;
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::pipe::MemoryOutputPipe;
use wasmtime_wasi::preview1::{self, WasiP1Ctx};

/// Default execution timeout in seconds for sandbox operations.
const DEFAULT_TIMEOUT_SECS: u64 = 30;
/// Number of epoch ticks allowed before the sandbox execution is interrupted.
/// One tick fires per second (see the background incrementer in `SandboxManager::new`),
/// so this acts as a coarse 1-second interrupt deadline independent of fuel.
const EPOCH_DEADLINE_TICKS: u64 = 1;

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub memory_limit_bytes: usize,
    pub cpu_fuel_limit: u64,
    /// Maximum wall-clock seconds before the sandbox execution is aborted.
    pub timeout_secs: u64,
}

struct SandboxState {
    wasi: WasiP1Ctx,
    limits: StoreLimits,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit_bytes: 256 * 1024 * 1024,
            cpu_fuel_limit: 100_000_000,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }
}

pub struct SandboxResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub struct SandboxManager {
    engine: Engine,
    config: SandboxConfig,
    _epoch_task: tokio::task::JoinHandle<()>,
}

impl SandboxManager {
    pub fn new(config: SandboxConfig) -> Result<Self> {
        anyhow::ensure!(
            config.memory_limit_bytes > 0,
            "SandboxConfig.memory_limit_bytes must be > 0"
        );
        anyhow::ensure!(
            config.cpu_fuel_limit > 0,
            "SandboxConfig.cpu_fuel_limit must be > 0"
        );
        anyhow::ensure!(
            config.timeout_secs > 0,
            "SandboxConfig.timeout_secs must be > 0"
        );
        let mut wasm_cfg = Config::new();
        wasm_cfg.consume_fuel(true);
        wasm_cfg.epoch_interruption(true);
        let engine = Engine::new(&wasm_cfg)?;

        // Start background epoch incrementer
        let engine_clone = engine.clone();
        let epoch_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                engine_clone.increment_epoch();
            }
        });

        Ok(Self {
            engine,
            config,
            _epoch_task: epoch_task,
        })
    }

    /// Execute WASM bytes synchronously (blocking). Prefer `execute_with_timeout`
    /// for async contexts to avoid hanging the executor.
    pub fn execute(&self, wasm_bytes: &[u8]) -> Result<SandboxResult> {
        let start = Instant::now();
        let mut linker = Linker::new(&self.engine);
        preview1::add_to_linker_sync(&mut linker, |s: &mut SandboxState| &mut s.wasi)?;

        let stdout_pipe = MemoryOutputPipe::new(1024 * 1024);
        let stderr_pipe = MemoryOutputPipe::new(1024 * 1024);

        let wasi = WasiCtxBuilder::new()
            .stdout(stdout_pipe.clone())
            .stderr(stderr_pipe.clone())
            .build_p1();

        let limits = StoreLimitsBuilder::new()
            .memory_size(self.config.memory_limit_bytes)
            .build();

        let mut store = Store::new(&self.engine, SandboxState { wasi, limits });
        store.limiter(|state| &mut state.limits);
        store.set_fuel(self.config.cpu_fuel_limit)?;
        store.set_epoch_deadline(EPOCH_DEADLINE_TICKS);

        let module = Module::from_binary(&self.engine, wasm_bytes)
            .context("failed to compile WASM module from provided bytes")?;
        linker
            .module(&mut store, "", &module)
            .context("failed to link WASM module")?;

        let func = linker
            .get_default(&mut store, "")
            .context("no default export in WASM module")?
            .typed::<(), ()>(&store)
            .context("default export has unexpected signature (expected () -> ())")?;

        let res = func.call(&mut store, ());
        let _fuel_consumed = store
            .get_fuel()
            .ok()
            .map(|f| self.config.cpu_fuel_limit - f);
        let _elapsed_ms = start.elapsed().as_millis() as u64;

        let exit_code = match res {
            Ok(_) => 0,
            Err(e) => {
                tracing::warn!("WASM execution failed: {e}");
                1
            }
        };

        let stdout = String::from_utf8_lossy(&stdout_pipe.contents()).into_owned();
        let stderr = String::from_utf8_lossy(&stderr_pipe.contents()).into_owned();

        Ok(SandboxResult {
            exit_code,
            stdout,
            stderr,
        })
    }

    /// Execute WASM bytes with a wall-clock timeout guard. The blocking WASM
    /// execution runs on the tokio blocking thread pool so it cannot stall the
    /// async reactor, and `tokio::time::timeout` enforces the deadline.
    pub async fn execute_with_timeout(
        self: &Arc<Self>,
        wasm_bytes: &[u8],
    ) -> Result<SandboxResult> {
        let timeout = tokio::time::Duration::from_secs(self.config.timeout_secs);
        let bytes = wasm_bytes.to_vec();
        let this = Arc::clone(self);

        let result = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || this.execute(&bytes)),
        )
        .await;

        match result {
            Ok(Ok(inner)) => inner,
            Ok(Err(join_err)) => Err(anyhow::anyhow!(
                "sandbox execution task panicked: {join_err}"
            )),
            Err(_elapsed) => Err(anyhow::anyhow!(
                "sandbox execution timed out after {}s",
                self.config.timeout_secs
            )),
        }
    }
}

impl Drop for SandboxManager {
    fn drop(&mut self) {
        self._epoch_task.abort();
    }
}
