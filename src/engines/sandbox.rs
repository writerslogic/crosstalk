use crate::types::conversation::ConversationState;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use wasmtime::*;
use wasmtime_wasi::pipe::MemoryOutputPipe;
use wasmtime_wasi::{ResourceTable, WasiCtx, WasiCtxBuilder, WasiView};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub struct SandboxConfig {
    pub memory_limit_bytes: usize,
    pub fuel_limit: u64,
}

struct MyState {
    wasi_ctx: WasiCtx,
    table: ResourceTable,
    stdout: MemoryOutputPipe,
    stderr: MemoryOutputPipe,
    resource_limiter: StoreLimits,
}

impl WasiView for MyState {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

impl MyState {
    fn new(mem_limit: usize) -> Self {
        let stdout = MemoryOutputPipe::new(1024 * 1024);
        let stderr = MemoryOutputPipe::new(1024 * 1024);
        let wasi_ctx = WasiCtxBuilder::new()
            .stdout(stdout.clone())
            .stderr(stderr.clone())
            .build();

        Self {
            wasi_ctx,
            table: ResourceTable::new(),
            stdout,
            stderr,
            resource_limiter: StoreLimitsBuilder::new().memory_size(mem_limit).build(),
        }
    }
}

pub struct SandboxManager {
    engine: Engine,
}

impl SandboxManager {
    pub fn new() -> Result<Self> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config)?;
        Ok(Self { engine })
    }

    pub async fn execute_with_rollback(
        &self,
        wasm_bytes: &[u8],
        config: &SandboxConfig,
        snapshot: &ConversationState,
    ) -> Result<(SandboxResult, Option<ConversationState>)> {
        match self.execute(wasm_bytes, config).await {
            Ok(result) if result.exit_code == 0 => Ok((result, None)),
            Ok(result) => Ok((result, Some(snapshot.clone()))),
            Err(e) => Ok((
                SandboxResult { exit_code: -1, stdout: String::new(), stderr: e.to_string() },
                Some(snapshot.clone()),
            )),
        }
    }

    pub async fn execute(
        &self,
        wasm_bytes: &[u8],
        config: &SandboxConfig,
    ) -> Result<SandboxResult> {
        let mut store = Store::new(&self.engine, MyState::new(config.memory_limit_bytes));
        store.set_fuel(config.fuel_limit)?;
        store.limiter(|s| &mut s.resource_limiter);

        let component = wasmtime::component::Component::from_binary(&self.engine, wasm_bytes)?;
        let mut linker: wasmtime::component::Linker<MyState> =
            wasmtime::component::Linker::new(&self.engine);
        wasmtime_wasi::add_to_linker_sync(&mut linker)?;

        let instance = linker.instantiate_async(&mut store, &component).await?;
        let run_func = instance.get_typed_func::<(), ()>(&mut store, "_start")?;

        let exit_code = run_func
            .call_async(&mut store, ())
            .await
            .map(|_| 0)
            .unwrap_or(-1);

        let stdout_bytes = store.data().stdout.contents();
        let stderr_bytes = store.data().stderr.contents();

        Ok(SandboxResult {
            exit_code,
            stdout: String::from_utf8_lossy(&stdout_bytes).to_string(),
            stderr: String::from_utf8_lossy(&stderr_bytes).to_string(),
        })
    }
}
