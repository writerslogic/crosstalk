//! Engine modules — each module owns one capability domain.
//! Re-exported types below form the stable internal API used by the orchestrator.
pub mod analytics;
pub mod collective_intelligence;
pub mod compute;
pub mod consensus;
pub mod diff;
pub mod intelligence;
pub mod linter;
pub mod memory;
pub mod planning;
pub mod proof;
pub mod quality;
pub mod reasoning;
pub mod release;
pub mod sandbox;
pub mod security;
pub mod self_improvement;
pub mod simulation;
pub mod surprise;
pub mod swarm;
pub mod validation;
pub mod verification;

// Re-export commonly-used types for ergonomic imports
pub use analytics::{AnalyticsEngine, FailureTaxonomy, QualityTrendDetector};
pub use consensus::{
    CertaintyAnalyzer, InfluenceWeightManager, KalmanConvergence, NashSolver, PayoffCalculator,
};
pub use intelligence::{IntelligenceEngine, QualityScorer};
pub use reasoning::{FallacyDetector, ReasoningEngine, ReasoningScorer, SynthesisEngine};
pub use validation::AstValidator;
pub use verification::{ContinuousAuditor, HashChain, InvariantChecker, ProofExporter};
