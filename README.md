# oxide-flux-runtime

> **The singularity point of the Flux→PTX stack.**
>
> One runtime. Five layers. Infinite GPUs.

`oxide-flux-runtime` is the top-level orchestrator for the Flux→PTX distributed GPU system. It is the single entry point through which Flux bytecode becomes persistent, warp-level GPU kernels—spanning compilation, state synchronization, fleet coordination, and bare-metal execution.

If you are building with Flux, you start **here**.

---

## What This Crate Is

Think of `oxide-flux-runtime` as the operating system kernel for a distributed GPU computer. It does not merely compile code; it manages the entire lifecycle of a Flux program across a heterogenous fleet of CUDA-capable nodes.

This crate combines **all five layers** of the Flux→PTX architecture into one coherent, stateful runtime:

| Layer | Responsibility | Crate |
|---|---|---|
| **Constructs** | Git-native GPU capabilities (skills, equipment) | [`oxide-constructs`](https://github.com/SuperInstance/oxide-constructs) |
| **Flux Compiler** | Bytecode → MIR → Pliron → PTX | [`flux-importer`](https://github.com/SuperInstance/flux-importer) |
| **Distributed State** | CRDT-based sync across nodes | [`oxide-crdt`](https://github.com/SuperInstance/oxide-crdt) |
| **Fleet Coordination** | Discovery, negotiation, rhythm | [`oxide-fleet`](https://github.com/SuperInstance/oxide-fleet) |
| **Execution** | Persistent kernels, warp-level consensus | [`cudaclaw-bridge`](https://github.com/SuperInstance/cudaclaw-bridge) |

The runtime is the **only** component that touches all five. It is the glue, the conductor, and the gatekeeper.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        OXIDE-FLUX-RUNTIME                               │
│                    "The entry point to everything"                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   CONSTRUCT LAYER   │  │   COMPILER LAYER    │  │    FLEET LAYER      │
│ Skills + Equipment  │  │ Bytecode → MIR →PTX │  │ Discovery, Rhythm   │
│   (git-native)      │  │  (flux-importer)    │  │   (oxide-fleet)     │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │                        │                        │
           │                        ▼                        │
           │           ┌─────────────────────┐               │
           │           │    CRDT LAYER       │               │
           │           │  Distributed State  │◄──────────────┘
           │           │   (oxide-crdt)      │
           │           └──────────┬──────────┘
           │                      │
           ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CUDACLAW EXECUTION LAYER                           │
│              Persistent Kernels · 400K+ ops/s · Warp Consensus          │
│                        (cudaclaw-bridge)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Five Layers, Explained

1. **Construct Layer (`oxide-constructs`)**  
   GPU-native capabilities are not libraries in the traditional sense. They are *constructs*—self-describing, git-addressable units of skill and equipment. A construct might be a ternary attention kernel, a custom reduction primitive, or a full agent personality. The runtime loads them on demand and verifies their identity before execution.

2. **Flux Compiler (`flux-importer`)**  
   Flux bytecode is a high-level, portable intermediate representation. The compiler pipeline lowers it through synthetic MIR, through the Pliron SSA infrastructure, and finally emits PTX that targets the node's specific compute capability. The runtime drives this pipeline and caches the result.

3. **Distributed State (`oxide-crdt`)**  
   GPUs are not islands. The runtime maintains CRDTs for kernel registries, agent states, and performance metrics. When a node joins the fleet, it does not ask "what is the state?"—it *converges* to it.

4. **Fleet Coordination (`oxide-fleet`)**  
   Agents discover each other, negotiate work distribution, and maintain a collective rhythm. The runtime exposes this as a first-class concern: your program does not run *on* a GPU; it runs *within* a fleet.

5. **cudaclaw Execution (`cudaclaw-bridge`)**  
   This is the metal. Persistent kernels that never return. Warp-level consensus primitives. 400,000+ operations per second on modern hardware. The runtime deploys compiled PTX into this substrate and manages its lifecycle.

---

## Runtime Lifecycle

A healthy runtime moves through six distinct phases. Each phase is observable, queryable, and recoverable.

```
    ┌─────┐     ┌──────────┐     ┌────────┐     ┌─────────┐     ┌────────┐     ┌──────────┐
    │ Init│────►│ Compile  │────►│ Deploy │────►│ Execute │────►│ Drain  │────►│ Shutdown │
    └─────┘     └──────────┘     └────────┘     └─────────┘     └────────┘     └──────────┘
       │                                                         ▲
       └─────────────────────────────────────────────────────────┘
                    (graceful recovery path before shutdown)
```

| Phase | `RuntimeStatus` | What Happens |
|---|---|---|
| **Init** | `Idle` | Runtime created, constructs empty, caches warm. |
| **Compile** | `Compiling { program }` | Bytecode validated against GPU requirements and construct dependencies. Lowered to PTX. Cached. |
| **Deploy** | `Deploying { program }` | PTX handed to `cudaclaw-bridge`. Kernel allocated on device. |
| **Execute** | `Executing { program, kernels_running }` | Kernel is live. Warp consensus begins. CRDT state syncs. Fleet coordination active. |
| **Drain** | `Draining` | Graceful stop. Active kernels finish in-flight work. No new programs accepted. |
| **Shutdown** | `Shutdown` | All kernels retired. VRAM released. Runtime terminates. |

---

## Quick Start

### Configure the Runtime

```rust
use oxide_flux_runtime::{OxideFluxRuntime, RuntimeConfig};

let config = RuntimeConfig {
    max_workers: 16,              // CUDA streams / worker threads
    total_vram_mb: 24_576,        // Treat this node as having 24 GB
    node_id: "gpu-node-3".to_string(),
    enable_crdt_sync: true,       // Join the distributed mesh
    enable_fleet: true,           // Negotiate with peer nodes
    compute_capability: 80,       // Target SM_80 (Ampere)
};

let mut runtime = OxideFluxRuntime::new(config);
```

### Load Constructs (Git-Native Capabilities)

Constructs are loaded by repository address. They are idempotent—loading the same construct twice is a no-op.

```rust
// Load a ternary attention kernel from the SuperInstance construct registry
runtime.load_construct("SuperInstance/ternary-attention")?;
runtime.load_construct("SuperInstance/fleet-rhythm-sync")?;

// Inspect what we have
assert_eq!(runtime.loaded_constructs().len(), 2);

// Unload when no longer needed (fails if active kernels depend on it)
runtime.unload_construct("SuperInstance/fleet-rhythm-sync")?;
```

In production, `load_construct` performs a shallow git clone, parses `CONSTRUCT.toml`, and verifies the construct's cryptographic identity before registering it.

### Define a Flux Program

```rust
use oxide_flux_runtime::{FluxProgram, GpuRequirements};

let program = FluxProgram {
    name: "distributed-attention-forward".to_string(),
    bytecode: vec![0x01, 0x00, 0x2A, 0x00, 0xFF], // Your Flux IR
    required_constructs: vec![
        "SuperInstance/ternary-attention".to_string(),
    ],
    gpu_requirements: GpuRequirements {
        min_compute_capability: 80,   // Needs Ampere or newer
        min_vram_mb: 4_096,           // 4 GB minimum
        block_dim: (256, 1, 1),       // One warp per block
        shared_mem_bytes: 48 * 1024,  // 48 KB shared memory
        uses_tensor_cores: true,      // Uses WMMA / MMA instructions
    },
};
```

### Compile and Execute

```rust
// Compile: bytecode → PTX (cached for reuse)
let ptx = runtime.compile(&program)?;
println!("Generated {} bytes of PTX", ptx.len());

// Execute: deploy kernel and run
let result = runtime.execute(&program)?;
println!("Kernel {} is {} on {}",
    result.kernel_id,
    result.status,
    result.gpu_node
);

// Or do both in one shot
let result = runtime.run(&program)?;
```

### Query Runtime State

```rust
// Current phase
match runtime.status() {
    RuntimeStatus::Idle => println!("Ready for work"),
    RuntimeStatus::Executing { program, kernels_running } => {
        println!("Running {} with {} active kernels", program, kernels_running);
    }
    _ => {}
}

// Cumulative statistics
let stats = runtime.stats();
println!("Compiled: {}", stats.programs_compiled);
println!("Executed: {}", stats.total_kernel_invocations);
println!("Fleet size: {} nodes", stats.fleet_size);
println!("Peak VRAM: {} MB", stats.peak_vram_used_mb);
```

### Graceful Shutdown

```rust
// Signal all kernels to finish in-flight work
runtime.shutdown();
assert_eq!(*runtime.status(), RuntimeStatus::Shutdown);
assert!(runtime.active_kernels().is_empty());
```

---

## Error Handling

The runtime is strict about what it will execute. There are no silent failures—every violation is a typed `RuntimeError`.

### `CapabilityMismatch`

Raised when a program requires a GPU feature newer than what the node provides.

```rust
let config = RuntimeConfig {
    compute_capability: 70, // This node is Volta
    ..Default::default()
};
let mut rt = OxideFluxRuntime::new(config);

let mut program = make_program("sm90-test");
program.gpu_requirements.min_compute_capability = 90; // Requires Hopper

assert!(matches!(
    rt.compile(&program).unwrap_err(),
    RuntimeError::CapabilityMismatch { required: 90, available: 70 }
));
```

### `MissingConstruct`

Raised when a program declares a dependency on a construct that has not been loaded.

```rust
let mut program = make_program("needs-ternary");
program.required_constructs = vec!["SuperInstance/missing-kernel".to_string()];

assert!(matches!(
    rt.compile(&program).unwrap_err(),
    RuntimeError::MissingConstruct(_)
));
```

### Other Errors

| Error | Cause |
|---|---|
| `CompilationFailed(String)` | The `flux-importer` pipeline produced invalid MIR or PTX. |
| `DeploymentFailed(String)` | `cudaclaw-bridge` rejected the kernel (invalid PTX, out of VRAM). |
| `NotReady` | The runtime is in `Draining` or `Shutdown` and cannot accept new work. |
| `AlreadyShutdown` | An operation was attempted after `shutdown()` has been called. |

---

## Relationship to the Ecosystem

`oxide-flux-runtime` does not replace the other crates—it **composes** them. Understanding the boundary is essential.

| Crate | Owned By Runtime? | Relationship |
|---|---|---|
| `flux-importer` | No | The runtime *drives* the compiler. It passes bytecode and receives PTX. It does not know about MIR or Pliron internals. |
| `oxide-constructs` | No | The runtime *loads* constructs. It validates their identity and checks their exports against program requirements. The construct format itself is owned by `oxide-constructs`. |
| `oxide-crdt` | Partially | The runtime initializes CRDT state and triggers syncs. The CRDT algorithms and wire format live in `oxide-crdt`. |
| `oxide-fleet` | Partially | The runtime enables or disables fleet participation via `enable_fleet`. Discovery and negotiation protocols are implemented in `oxide-fleet`. |
| `cudaclaw-bridge` | No | The runtime *deploys* to cudaclaw. Persistent kernel management, warp consensus, and the actual CUDA driver calls are the bridge's domain. |

In short: **the runtime is the API surface; the other crates are the engine.**

---

## Design Philosophy

> "A GPU is not a co-processor. It is a first-class node in a distributed system."

The runtime is built on three principles:

1. **Git-native everything.** Constructs are identified by git coordinates, not opaque package IDs. Reproducibility is built in.

2. **Fail fast at the boundary.** Capability mismatches and missing constructs are caught at *compile time* (before a single byte hits VRAM), not at kernel launch.

3. **Observability is not optional.** Every phase of the lifecycle is reflected in `RuntimeStatus` and `RuntimeStats`. There is no hidden state.

---

## License

Apache-2.0

---

## See Also

- [`flux-importer`](https://github.com/SuperInstance/flux-importer) — The Flux→MIR→PTX compiler pipeline
- [`oxide-constructs`](https://github.com/SuperInstance/oxide-constructs) — Git-native GPU capability registry
- [`oxide-crdt`](https://github.com/SuperInstance/oxide-crdt) — Distributed state synchronization
- [`oxide-fleet`](https://github.com/SuperInstance/oxide-fleet) — Fleet discovery and work negotiation
- [`cudaclaw-bridge`](https://github.com/SuperInstance/cudaclaw-bridge) — Persistent kernel execution substrate
