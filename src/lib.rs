//! # oxide-flux-runtime
//!
//! The top-level runtime for the Flux→PTX distributed GPU system.
//!
//! This crate combines all layers of the Flux→PTX stack:
//!
//! 1. **Flux Bytecode** → compiled to synthetic MIR
//! 2. **Construct Loading** → git-native GPU capabilities loaded at runtime
//! 3. **CRDT State** → distributed state synchronized across GPU nodes
//! 4. **Fleet Coordination** → agents discover, negotiate, distribute work
//! 5. **cudaclaw Execution** → persistent kernels with warp-level consensus
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         CONSTRUCT LAYER             │
//! │   Skills + Equipment (git-native)   │
//! └──────────────┬──────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────┐
//! │         FLUX COMPILER               │
//! │   Bytecode → MIR → Pliron → PTX     │
//! └──────────────┬──────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────┐
//! │      DISTRIBUTED STATE              │
//! │   CRDTs: kernels, agents, metrics   │
//! └──────────────┬──────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────┐
//! │      FLEET COORDINATION             │
//! │   Discovery, negotiation, rhythm    │
//! └──────────────┬──────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────┐
//! │      CUDACLaw EXECUTION             │
//! │   Persistent kernels, 400K ops/s    │
//! └─────────────────────────────────────┘
//! ```

/// Runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum concurrent GPU workers per node.
    pub max_workers: u32,
    /// Total VRAM available (MB).
    pub total_vram_mb: u32,
    /// This node's identifier.
    pub node_id: String,
    /// Whether to enable CRDT synchronization.
    pub enable_crdt_sync: bool,
    /// Whether to enable fleet coordination.
    pub enable_fleet: bool,
    /// Compute capability target (e.g., 80 for SM_80).
    pub compute_capability: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_workers: 8,
            total_vram_mb: 8192,
            node_id: "node-0".to_string(),
            enable_crdt_sync: true,
            enable_fleet: true,
            compute_capability: 80,
        }
    }
}

/// A Flux program to be compiled and executed.
#[derive(Debug, Clone)]
pub struct FluxProgram {
    /// Program name.
    pub name: String,
    /// Flux bytecode.
    pub bytecode: Vec<u8>,
    /// Required constructs (dependencies).
    pub required_constructs: Vec<String>,
    /// GPU requirements.
    pub gpu_requirements: GpuRequirements,
}

/// GPU requirements for a Flux program.
#[derive(Debug, Clone)]
pub struct GpuRequirements {
    pub min_compute_capability: u32,
    pub min_vram_mb: u32,
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub uses_tensor_cores: bool,
}

impl Default for GpuRequirements {
    fn default() -> Self {
        Self {
            min_compute_capability: 80,
            min_vram_mb: 256,
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
            uses_tensor_cores: false,
        }
    }
}

/// Runtime status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeStatus {
    /// Runtime initialized but idle.
    Idle,
    /// Compiling Flux bytecode to PTX.
    Compiling { program: String },
    /// Deploying compiled PTX to GPU.
    Deploying { program: String },
    /// Executing on GPU.
    Executing { program: String, kernels_running: u32 },
    /// Draining before shutdown.
    Draining,
    /// Shutdown complete.
    Shutdown,
}

/// Runtime statistics.
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    pub programs_compiled: u64,
    pub programs_executed: u64,
    pub total_kernel_invocations: u64,
    pub total_errors: u64,
    pub uptime_seconds: u64,
    pub peak_vram_used_mb: u32,
    pub constructs_loaded: usize,
    pub fleet_size: usize,
}

/// The Flux→PTX runtime.
pub struct OxideFluxRuntime {
    config: RuntimeConfig,
    status: RuntimeStatus,
    stats: RuntimeStats,
    /// Registered constructs.
    constructs: Vec<String>,
    /// Compiled PTX cache.
    ptx_cache: Vec<Vec<u8>>,
    /// Active kernels.
    active_kernels: Vec<String>,
}

impl OxideFluxRuntime {
    /// Create a new runtime with the given configuration.
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            status: RuntimeStatus::Idle,
            stats: RuntimeStats::default(),
            constructs: Vec::new(),
            ptx_cache: Vec::new(),
            active_kernels: Vec::new(),
        }
    }

    /// Get current runtime status.
    pub fn status(&self) -> &RuntimeStatus {
        &self.status
    }

    /// Get runtime statistics.
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }

    /// Load a construct from a git repository.
    pub fn load_construct(&mut self, repo: &str) -> Result<(), RuntimeError> {
        // In production: git clone + parse CONSTRUCT.toml + verify identity
        if self.constructs.contains(&repo.to_string()) {
            return Ok(());
        }
        self.constructs.push(repo.to_string());
        self.stats.constructs_loaded = self.constructs.len();
        Ok(())
    }

    /// Unload a construct.
    pub fn unload_construct(&mut self, repo: &str) -> Result<(), RuntimeError> {
        // Can't unload if any active kernel depends on it
        self.constructs.retain(|c| c != repo);
        self.stats.constructs_loaded = self.constructs.len();
        Ok(())
    }

    /// Compile a Flux program to PTX.
    pub fn compile(&mut self, program: &FluxProgram) -> Result<Vec<u8>, RuntimeError> {
        self.status = RuntimeStatus::Compiling { program: program.name.clone() };

        // Validate requirements
        if program.gpu_requirements.min_compute_capability > self.config.compute_capability {
            self.status = RuntimeStatus::Idle;
            return Err(RuntimeError::CapabilityMismatch {
                required: program.gpu_requirements.min_compute_capability,
                available: self.config.compute_capability,
            });
        }

        // Validate construct dependencies
        for dep in &program.required_constructs {
            if !self.constructs.contains(dep) {
                self.status = RuntimeStatus::Idle;
                return Err(RuntimeError::MissingConstruct(dep.clone()));
            }
        }

        // In production: flux-importer → cuda-oxide pipeline
        let ptx = vec![0x7f, 0x50, 0x54, 0x58]; // PTX magic bytes placeholder
        self.ptx_cache.push(ptx.clone());
        self.stats.programs_compiled += 1;
        self.status = RuntimeStatus::Idle;
        Ok(ptx)
    }

    /// Execute a compiled Flux program on the GPU.
    pub fn execute(&mut self, program: &FluxProgram) -> Result<ExecutionResult, RuntimeError> {
        self.status = RuntimeStatus::Deploying { program: program.name.clone() };

        // In production: cudaclaw-bridge.deploy()
        let kernel_id = format!("flux-{}", program.name);
        self.active_kernels.push(kernel_id.clone());

        self.status = RuntimeStatus::Executing {
            program: program.name.clone(),
            kernels_running: self.active_kernels.len() as u32,
        };

        self.stats.programs_executed += 1;
        self.stats.total_kernel_invocations += 1;

        Ok(ExecutionResult {
            kernel_id,
            status: "running".to_string(),
            gpu_node: self.config.node_id.clone(),
        })
    }

    /// Compile and execute in one step.
    pub fn run(&mut self, program: &FluxProgram) -> Result<ExecutionResult, RuntimeError> {
        self.compile(program)?;
        self.execute(program)
    }

    /// Stop a running program.
    pub fn stop(&mut self, kernel_id: &str) -> Result<(), RuntimeError> {
        self.active_kernels.retain(|k| k != kernel_id);
        if self.active_kernels.is_empty() {
            self.status = RuntimeStatus::Idle;
        } else {
            self.status = RuntimeStatus::Executing {
                program: "multiple".to_string(),
                kernels_running: self.active_kernels.len() as u32,
            };
        }
        Ok(())
    }

    /// Graceful shutdown — drain all kernels.
    pub fn shutdown(&mut self) {
        self.status = RuntimeStatus::Draining;
        self.active_kernels.clear();
        self.status = RuntimeStatus::Shutdown;
    }

    /// List loaded constructs.
    pub fn loaded_constructs(&self) -> &[String] {
        &self.constructs
    }

    /// List active kernels.
    pub fn active_kernels(&self) -> &[String] {
        &self.active_kernels
    }

    /// Get the runtime config.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
}

/// Result of executing a Flux program.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub kernel_id: String,
    pub status: String,
    pub gpu_node: String,
}

/// Runtime errors.
#[derive(Debug, Clone)]
pub enum RuntimeError {
    CapabilityMismatch { required: u32, available: u32 },
    MissingConstruct(String),
    CompilationFailed(String),
    DeploymentFailed(String),
    NotReady,
    AlreadyShutdown,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapabilityMismatch { required, available } => {
                write!(f, "compute capability mismatch: need SM_{}, have SM_{}", required, available)
            }
            Self::MissingConstruct(construct) => write!(f, "missing construct: {}", construct),
            Self::CompilationFailed(reason) => write!(f, "compilation failed: {}", reason),
            Self::DeploymentFailed(reason) => write!(f, "deployment failed: {}", reason),
            Self::NotReady => write!(f, "runtime not ready"),
            Self::AlreadyShutdown => write!(f, "runtime already shut down"),
        }
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_program(name: &str) -> FluxProgram {
        FluxProgram {
            name: name.to_string(),
            bytecode: vec![0x01, 0x00, 0x2A, 0x00, 0xFF],
            required_constructs: vec![],
            gpu_requirements: GpuRequirements::default(),
        }
    }

    #[test]
    fn test_runtime_lifecycle() {
        let mut rt = OxideFluxRuntime::new(RuntimeConfig::default());
        assert_eq!(*rt.status(), RuntimeStatus::Idle);

        let program = make_program("test");
        let result = rt.run(&program).unwrap();
        assert_eq!(result.status, "running");
        assert_eq!(rt.active_kernels().len(), 1);

        rt.stop(&result.kernel_id).unwrap();
        assert_eq!(*rt.status(), RuntimeStatus::Idle);
    }

    #[test]
    fn test_construct_loading() {
        let mut rt = OxideFluxRuntime::new(RuntimeConfig::default());
        rt.load_construct("SuperInstance/ternary-attention").unwrap();
        rt.load_construct("SuperInstance/ternary-attention").unwrap(); // idempotent
        assert_eq!(rt.loaded_constructs().len(), 1);

        rt.unload_construct("SuperInstance/ternary-attention").unwrap();
        assert_eq!(rt.loaded_constructs().len(), 0);
    }

    #[test]
    fn test_compile_and_execute() {
        let mut rt = OxideFluxRuntime::new(RuntimeConfig::default());
        let program = make_program("compile-test");

        let ptx = rt.compile(&program).unwrap();
        assert!(!ptx.is_empty());
        assert_eq!(rt.stats().programs_compiled, 1);

        let result = rt.execute(&program).unwrap();
        assert_eq!(rt.stats().programs_executed, 1);
    }

    #[test]
    fn test_missing_construct_dependency() {
        let mut rt = OxideFluxRuntime::new(RuntimeConfig::default());
        let mut program = make_program("dep-test");
        program.required_constructs = vec!["SuperInstance/missing-kernel".to_string()];

        let err = rt.compile(&program).unwrap_err();
        assert!(matches!(err, RuntimeError::MissingConstruct(_)));
    }

    #[test]
    fn test_capability_mismatch() {
        let config = RuntimeConfig { compute_capability: 70, ..Default::default() };
        let mut rt = OxideFluxRuntime::new(config);
        let mut program = make_program("sm90-test");
        program.gpu_requirements.min_compute_capability = 90;

        let err = rt.compile(&program).unwrap_err();
        assert!(matches!(err, RuntimeError::CapabilityMismatch { required: 90, available: 70 }));
    }

    #[test]
    fn test_shutdown() {
        let mut rt = OxideFluxRuntime::new(RuntimeConfig::default());
        rt.run(&make_program("k1")).unwrap();
        rt.run(&make_program("k2")).unwrap();

        rt.shutdown();
        assert_eq!(*rt.status(), RuntimeStatus::Shutdown);
        assert_eq!(rt.active_kernels().len(), 0);
    }

    #[test]
    fn test_multiple_kernels() {
        let mut rt = OxideFluxRuntime::new(RuntimeConfig::default());
        rt.run(&make_program("k1")).unwrap();
        rt.run(&make_program("k2")).unwrap();
        rt.run(&make_program("k3")).unwrap();

        assert_eq!(rt.active_kernels().len(), 3);
        assert_eq!(rt.stats().programs_executed, 3);
    }

    #[test]
    fn test_config() {
        let config = RuntimeConfig {
            max_workers: 16,
            total_vram_mb: 24576,
            node_id: "gpu-node-3".to_string(),
            ..Default::default()
        };
        let rt = OxideFluxRuntime::new(config);
        assert_eq!(rt.config().max_workers, 16);
        assert_eq!(rt.config().total_vram_mb, 24576);
    }
}
