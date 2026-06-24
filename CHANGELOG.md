# Changelog

All notable changes to this project are generated from the commit history.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) +
[Conventional Commits](https://www.conventionalcommits.org/).
## [Unreleased]

### Added
- Runnable cross-verification of the cogmem C2PA sample
- Emit orchestration audit statement into the live turn loop
- Emit orchestration audit as shared-substrate COSE SCITT signed statement
- Config file support and provider-specific error URLs
- First-run API key setup wizard and help overlay
- Intelligence pipeline improvements and self-evolution
- Wire DataMinimizer to session finalization, remove debug println
- Improve self-evolution and root cause analysis
- Improve intelligence pipeline, caching, and output quality
- Resolve all remaining open issues — complete fix-batch pass
- Expand test coverage and engine improvements
- Implement 10 intelligence upgrades
- Finalize high-integrity upgrades for Tracks 05-15 (Consensus, Sandbox, Env, Verification, Memory, Intel, Compute, Reasoning, Quality, Self-Improvement, Swarm)
- Certainty-weighted synthesis, turn compression, artifact health feedback, outlier detection, regression skip
- Wire agent routing, convergence tiers, surprise-calibrated weights, Thompson Sampling
- Wire NixManager into orchestrator, gateway, linter; additive env injection
- SkillProgressionTracker, CapabilityGapScanner, ReviewerCalibrator, UCB1ProtocolSelector, RoleSequenceRecorder
- PromptEvolutionaryOptimizer, CalibrationAdjuster, BenchmarkRegressionGuard, PostMortemLearner, RuntimeParameterAdjuster, ProgressReporter, LearningEffectivenessMonitor, EscalationContextBuilder, SelfCodeModifier::verify
- QualityTrendDetector, DecisionReplay, StrategyRecommender, MetaLearning, HomebrewFormula, DynamicTeamComposer, MetaStrategyOptimizer, TimelineManager, ReplayEngine, SvgExporter, 29 new tests
- BudgetManager, BatchScheduler, LatencyRouter, StrategyMixer, StructureSelector, FalseDichotomy+StrawMan, AssumptionExtractor, CrossExaminer, ArgumentParser, ReportGenerator, 4-dim ReasoningScorer, 20 new tests
- TaskDecomposer, AgentAssigner, ConflictDetector, ProgressMonitor, SwarmTelemetry, GoalTree methods, DifficultyEstimator, GoalScheduler, CriticalPathAnalyzer, MilestoneDetector, SessionManager
- Adaptive K selection, decay calibration, prompt mutation, convergence velocity tracker, Pareto optimizer
- Entropy heatmap, 4-pane layout, Ctrl+I fully wired, 60fps FPS counter, Tab/g/G navigation
- RefinementRound, AstVersionHistory, CLI bridges, ProofExporter, SessionContext, PromptComposer, dim-aware MemoryStore, fix all clippy warnings
- MemoryBridge with cross-session recall, SessionContext, 11 new tests
- SurpriseEngine with prediction recording, calibration, and 11 new tests
- NixManager, ModelEnsemble, LatencyPredictor, AuditAlert channel
- Implement regression detection for Track 10
- Add outcome-weighted retrieval for Track 09
- Add constraint-based task routing with budget and latency constraints
- Implement embedding pipeline for Track 09 with deterministic hashing and cosine similarity
- 3D latent space explorer, interaction graphs, and scrub timeline
- Agent specialization, knowledge transfer, and peer review
- Release manager, stability audit, and CPOP verification
- Convergence diagnostics, agent profiling, and failure taxonomy
- Secret scanning, shell sanity, and turn signing
- Goal tree management, hibernation, and context pruning
- Sub-orchestrator lifecycle, leader election, and sigma-syncing
- Self-evaluation, A/B testing, and self-code modification
- Artifact metrics, duplication detection, and regression blocking
- Adaptive turn structure, fallacy detection, and signal extraction
- Budget management, parallel inference, and resource monitoring
- Model profiling, task routing, and quality scoring
- Outcome-aware vector memory with LanceDB and context distillation
- Hash-chain audit, tautology filtering, and proof-carrying artifacts
- MCP hub with tool discovery and CLI bridges
- Sandboxed execution with Monte Carlo prediction and AST versioning
- Quantitative convergence with Kalman Filter and Nash Solver
- High-fidelity TUI dashboard with Ghost-Stream and Neural Intercept
- AST validation for Rust deltas
- Iteration tracking and Δα capture
- Character-level unified diffing
- Implement turn logic, rewind, and resume with integration tests
- Implement StateManager with Sled persistence and tests
- Fix rig dyn-compatibility and implement ConversationState types

### Changed
- Drop unread profiles parameter from select_adaptive
- Extract persist_snapshot helper for cross-session records
- Drop unused cache/executor/sharded from crosstalk-concurrency
- Remove dead bridge API, keep validated CliBridge::call
- Release sigma lock before finalize_session persistence
- Split orchestrator.rs (4681 lines) into orchestrator/ submodules
- Remove dead TieredMemoryManager instead of leaving it unwired
- Remove unreachable GodView GPU rendering and wgpu/winit/bytemuck deps
- Extract logging and background loop from main and drain on shutdown
- Reorganize into modular architecture (core, engines, types, mcp, ui, utils)

### Documentation
- Restructure README with collapsible sections
- Rewrite README — fix logo tag, add install/quick start, improve structure
- Polish README and add agent-provenance stack cross-reference
- Rewrite README and add writerslogic conventions (badges, community-health, dotfiles); set copyright to WritersLogic, Inc.

### Fixed
- Surface sandbox fuel/elapsed and flag resource-limit kills
- Propagate swallowed mode-transition emit errors
- Persist computed nix_env instead of discarding it
- Document guarded unwrap invariants with expect()
- Log workspace-escape file cleanup failure instead of swallowing
- Restore wiped crosstalk-concurrency primitives and wire as dependency
- Iteration quality, validation timeout, timeout-based agent skip
- Native provider routing, OpenRouter fallback, swarm resilience
- Memory caps, render perf, swallowed errors across 5 files
- Resolve all remaining open issues and test compilation failures
- Resolve all pre-existing clippy warnings across 7 files
- Path traversal, timeout, output cap, assumption cap, atomic migration
- 9 medium bugs from todo batch
- Resolve 3 critical bugs
- Apply review suggestions and formatting fixes for memory logic
- Use ort load-dynamic with catch_unwind for CoreML; release build succeeds
- Handle broadcast Lagged in swarm worker, validate project root in FileWriter
- Recover from mutex poisoning in TUI, propagate checkpoint channel death via save_all
- Resolve CRIT-001 CRIT-002 CRIT-003 silent errors and path traversal
- Lower tautology threshold 0.95 to 0.85 to detect similar paraphrases
- Remove orphaned StructureEnforcer test, fix zero-variance significance edge case
- Drain event channel in make_orchestrator to prevent closed-channel errors
- HIGH-015 add spawn_blocking wrappers for StateManager hot paths
- Update environment_tests and orchestrator_tests for current APIs
- Propagate silent failures in ctrl_tx, checkpoint scan, BranchRegistry
- CRITICAL-001 propagate errors in write_cache_entry helper
- Propagate event channel failures via emit() helper, stop silent swallowing
- Add per-model RequestRateLimiter with sliding window token bucket
- Store supervisor JoinHandle in SwarmController, add shutdown()
- Consolidate event loop lock acquisitions from 4 to 2 per iteration
- Async embed_text in recall_relevant via spawn_blocking
- 19 maintainability and quality improvements (M-004 to M-052)
- Convert write_artifact from std::fs to tokio::fs throughout
- Cache compiled regexes in StructureTemplate, log fastembed init failure
- Add SIGTERM/SIGINT handler to TUI render loop
- 9 HIGH bugs -- sh ext, restore propagate, panic hook order, size limit, key zeroize, temp collision, semaphore panic, API key in URL, LanceDB unwraps
- Symlink TOCTOU ordering, build.rs blocking, Cargo.toml/.github blocklist
- Gateway deadlock, linter blocking, query injection, orchestrator lock scope
- Symlink escape TOCTOU and Nix injection
- 8 critical/high bugs — orphan check >, ContinuousAuditor hash anchor, Voting tie-break, unwrap→match in get_node_name, diff_nodes equal-turn guard, record_snapshot dedup, linter path UTF-8, surprise certainty validation
- Update self_code_modifier test to use remaining safe pattern
- Address all 9 audit findings — t-critical, TrimVerbose chars, per-category budgets, StrategyEntry dimension padding, ETA guard, deterministic trend ordering, NaN filter, safe PATTERNS, RuntimeParameterAdjuster unknown param
- Wire recommendations into AnalyticsReport, SHA-256 cache keys, LatencyRouter::select(), per-agent StructureSelector, CI pipeline with clippy gate, StuckGoal/CapabilityMismatch/ConflictingProposals blockers
- Serialize snapshot env-var tests with static tokio mutex
- Batch_to_records dim bounds check, consensus convergence tracking, NaN-safe cmp, checkpoint temp cleanup, git log validation, render error context, embedding iter copy
- Unwrap HashChain::compute Result in auditor_tests
- Add await and workspace_dir to LinterGuard::check call
- Grant agent Full permissions in test_mcp_list_tools
- Resolve all 32 clippy violations in refactored codebase
- Repair broken tests, fix diff engine round-trip bug, add missing type definitions

### Performance
- Share session records via Arc to drop recall clones and divergent dual-writes (H-036, H-047)
- Stop serializing MCP tool calls on the gateway lock

### Security
- Add transcript hash chain anchored to git for keyless tamper-evidence
- Pin signing public key and verify turns against it, not the secret
- Encrypt signing seed at rest with passphrase-derived AEAD
- Persist signing key, verify turns on resume, wire audit/risk/injection defenses
- Confine shell_exec tool directives to the workspace
- Resolve all critical compound risks (CLU-001, CLU-002, CLU-006)

### Conductor
- Checkpoint end of Phase 1

### Deep-fix
- Category-specific regression baseline, word-boundary evidence detection, ArtifactDiff validation

### Improve
- Split CI tests into lib/integration, fix H-027 verify error handling, improve M-041
- Simplify M-025 and M-004 logic, add deny.toml for cargo-deny
- Symlink canonicalization throughout lifecycle and Nix validation at construction

### Style
- Add rustfmt.toml and normalize formatting
- Collapse nested if into single condition per clippy
- Format verification_tests.rs with rustfmt
- Format code with cargo fmt

### Task
- Fix all clippy warnings clean (P0)
- P3 MEDIUM error-handling fixes batch
- P3 MEDIUM performance fixes batch
- P3 MEDIUM architecture fixes batch
- P3 MEDIUM security fixes batch
- Sweep swallowed errors with ? propagation (SYS-002) batch 1
- Eliminate remaining clone-in-hot-path sites (SYS-003)
- Fix blocking-in-async sites (SYS-004)
- Fix DashMap sessions TOCTOU via atomic entry API (H-047)
- Redesign MCP gateway to &self interior sharding (H-019)
- Wire CancelScope into background orchestrator cancellation (H-040)
- Replace magic values with consts module (SYS-001)
- Minimize Sigma lock critical section (H-042)
- Introduce unified CrossTalkError type
- Route embed_text() through shared Cache (H-033)
- Migrate background orchestrator spawns to CancelScope (H-040)
- Write Sharded contract tests
- Write AsyncExecutor contract tests
- Write Cache contract tests
- Write CancelScope contract tests
- Implement Sharded<K,V> bounded sharded map
- Implement AsyncExecutor handle with thread-typed handles
- Implement moka-backed async Cache<K,V>
- Implement CancelScope primitive
- Validate Sharded access patterns against both call sites
- Scaffold crosstalk-concurrency internal crate
- Generate verifiable issue inventory

### Wip
- Orchestrator, compute, quality, swarm updates

### Wire
- Add ContinuousAuditor to runtime for continuous hash chain verification

