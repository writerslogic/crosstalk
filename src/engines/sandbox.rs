use anyhow::Result;
use wasmtime::*;
use wasmtime_wasi::sync::WasiCtxBuilder;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub memory_limit_bytes: usize,
    pub cpu_fuel_limit: u64,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit_bytes: 256 * 1024 * 1024, // 256MB
            cpu_fuel_limit: 100_000_000,           // 100M units
        }
    }
}

pub struct SandboxResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_ms: u64,
    pub fuel_consumed: Option<u64>,
}

pub struct SandboxManager {
    engine: Engine,
    config: SandboxConfig,
}

impl SandboxManager {
    pub fn new(config: SandboxConfig) -> Result<Self> {
        let mut wasm_cfg = Config::new();
        wasm_cfg.consume_fuel(true);
        let engine = Engine::new(&wasm_cfg)?;
        Ok(Self { engine, config })
    }

    pub fn execute(&self, wasm_bytes: &[u8]) -> Result<SandboxResult> {
        let start = Instant::now();
        let mut linker = Linker::new(&self.engine);
        wasmtime_wasi::add_to_linker(&mut linker, |s| s)?;

        let wasi = WasiCtxBuilder::new()
            .inherit_stdout()
            .inherit_stderr()
            .build();

        let mut store = Store::new(&self.engine, wasi);
        store.set_fuel(self.config.cpu_fuel_limit)?;

        let module = Module::from_binary(&self.engine, wasm_bytes)?;
        linker.module(&mut store, "", &module)?;

        let func = linker
            .get_default(&mut store, "")?
            .typed::<(), ()>(&store)?;

        let res = func.call(&mut store, ());
        let fuel_consumed = store.get_fuel().ok().map(|f| self.config.cpu_fuel_limit - f);
        
        let exit_code = match res {
            Ok(_) => 0,
            Err(e) => {
                if let Some(trap) = e.downcast_ref::<Trap>() {
                    match trap {
                        Trap::OutOfFuel => 137,
                        _ => 1,
                    }
                } else {
                    1
                }
            }
        };

        Ok(SandboxResult {
            exit_code,
            stdout: String::new(),
            stderr: String::new(),
            elapsed_ms: start.elapsed().as_millis() as u64,
            fuel_consumed,
        })
    }
}
