//! CrosstalkHarness: mock orchestrator implementing UCB1 topology selection.
//!
//! Mirrors the algorithm in `crosstalk::engines::topology::TopologyManager` without
//! requiring a live LLM backend, enabling fast, reproducible benchmarking.
//!
//! # Efficiency formula
//!
//! ```text
//! norm_cost    = (cost_usd   + ε_c) / (global_cost_mean   + ε_c)
//! norm_latency = (latency_ms + ε_l) / (global_latency_mean + ε_l)
//! efficiency   = quality / (α·norm_cost + β·norm_latency + ε_floor)
//! score_t      = efficiency_t + global_μ_eff · √(2·ln(N) / n_t)
//! ```
//!
//! # Budget mode weights
//!
//! | Mode          | α    | β   |
//! |---------------|------|-----|
//! | Normal        | 0.30 | 0.20 |
//! | CostReduction | 0.70 | 0.30 |
//! | Emergency     | 1.50 | 0.50 |

use rand::{
    Rng, SeedableRng,
    distr::{Bernoulli, Distribution},
    rngs::StdRng,
};
use std::collections::HashMap;

// ─── Domain types (mirror crosstalk::types) ────────────────────────────────

/// Debate topology variants.
///
/// Mirrors `crosstalk::engines::topology::DebateTopology`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DebateTopology {
    RoundRobin,
    Adversarial,
    Ensemble,
    TreeOfThoughts,
    Mediated,
    Critique,
}

impl DebateTopology {
    pub const ALL: [Self; 6] = [
        Self::RoundRobin,
        Self::Adversarial,
        Self::Ensemble,
        Self::TreeOfThoughts,
        Self::Mediated,
        Self::Critique,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::RoundRobin => "RoundRobin",
            Self::Adversarial => "Adversarial",
            Self::Ensemble => "Ensemble",
            Self::TreeOfThoughts => "TreeOfThoughts",
            Self::Mediated => "Mediated",
            Self::Critique => "Critique",
        }
    }

    /// Bernoulli probability of a correct answer on a math reasoning task.
    ///
    /// Values derived from the Crosstalk pilot study Table 2 (GSM8K, 3-agent runs).
    fn quality_prob(self) -> f64 {
        match self {
            Self::RoundRobin => 0.68,
            Self::Adversarial => 0.75,
            Self::Ensemble => 0.80,
            Self::TreeOfThoughts => 0.85,
            Self::Mediated => 0.73,
            Self::Critique => 0.70,
        }
    }

    /// Mean API cost in USD for a single turn (3-agent, ~1 k-token problem).
    fn mean_cost_usd(self) -> f64 {
        match self {
            Self::RoundRobin => 0.012,
            Self::Adversarial => 0.025,
            Self::Ensemble => 0.030,
            Self::TreeOfThoughts => 0.050,
            Self::Mediated => 0.028,
            Self::Critique => 0.015,
        }
    }

    /// Mean end-to-end wall-clock latency in milliseconds.
    fn mean_latency_ms(self) -> f64 {
        match self {
            Self::RoundRobin => 3_000.0,
            Self::Adversarial => 4_500.0,
            Self::Ensemble => 5_000.0,
            Self::TreeOfThoughts => 8_000.0,
            Self::Mediated => 5_500.0,
            Self::Critique => 3_500.0,
        }
    }

    /// Coefficient of variation for cost (log-normal noise).
    fn cost_cv(self) -> f64 {
        match self {
            Self::TreeOfThoughts => 0.20, // branch pruning causes higher variance
            _ => 0.15,
        }
    }

    /// Coefficient of variation for latency (log-normal noise).
    fn latency_cv(self) -> f64 {
        0.20
    }
}

/// Session budget mode.
///
/// Mirrors `crosstalk::types::compute::BudgetMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BudgetMode {
    /// >20 % of session budget remaining; balanced weights.
    Normal,
    /// 5–20 % remaining; cost penalty elevated.
    CostReduction,
    /// <5 % remaining; extreme cost and latency penalty.
    Emergency,
}

impl BudgetMode {
    /// Cost weight α in the efficiency denominator.
    pub fn alpha(self) -> f64 {
        match self {
            Self::Normal => 0.3,
            Self::CostReduction => 0.7,
            Self::Emergency => 1.5,
        }
    }

    /// Latency weight β in the efficiency denominator.
    pub fn beta(self) -> f64 {
        match self {
            Self::Normal => 0.2,
            Self::CostReduction => 0.3,
            Self::Emergency => 0.5,
        }
    }
}

// ─── UCB1 internal state ───────────────────────────────────────────────────

/// Welford's online running average (numerically stable mean + variance).
#[derive(Debug, Clone, Default)]
struct RunningAverage {
    pub mean: f64,
    pub count: u32,
    m2: f64, // variance accumulator
}

impl RunningAverage {
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
}

/// Per-topology running averages tracked by the UCB1 bandit.
#[derive(Debug, Clone, Default)]
struct TopologyStats {
    quality: RunningAverage,
    cost: RunningAverage,
    latency: RunningAverage,
}

// ─── Public harness ────────────────────────────────────────────────────────

/// Mock orchestrator harness implementing UCB1 topology selection.
///
/// Call [`CrosstalkHarness::run`] in a loop to simulate sequential task
/// execution. The harness maintains per-topology UCB1 state across calls and
/// updates it after every simulated outcome.
pub struct CrosstalkHarness {
    rng: StdRng,
    stats: HashMap<DebateTopology, TopologyStats>,
    total_turns: u64,
    global_cost: RunningAverage,
    global_latency: RunningAverage,
    global_efficiency: RunningAverage,
}

impl CrosstalkHarness {
    /// Create a harness seeded for reproducible runs.
    pub fn new(seed: u64) -> Self {
        let mut stats = HashMap::new();
        for t in DebateTopology::ALL {
            stats.insert(t, TopologyStats::default());
        }
        Self {
            rng: StdRng::seed_from_u64(seed),
            stats,
            total_turns: 0,
            global_cost: RunningAverage::default(),
            global_latency: RunningAverage::default(),
            global_efficiency: RunningAverage::default(),
        }
    }

    /// Simulate one orchestrator turn and return its outcome.
    ///
    /// Steps:
    /// 1. UCB1 selects the topology with the highest score (exploration-first).
    /// 2. Mock agent produces a probabilistic outcome via log-normal cost/latency
    ///    and Bernoulli quality sampling.
    /// 3. Running averages are updated for the next selection round.
    pub fn run(&mut self, mode: BudgetMode) -> RunResult {
        let topology = self.select_topology(mode);

        let (is_correct, latency_ms, cost_usd) = self.simulate_outcome(topology);
        let quality = if is_correct { 1.0 } else { 0.0 };

        // Compute efficiency using global means *before* this turn's update
        // (consistent with online UCB1: the selector sees past data only).
        let efficiency = self.compute_efficiency(quality, cost_usd, latency_ms, mode);

        // Update per-topology and global running averages.
        let s = self
            .stats
            .get_mut(&topology)
            .expect("topology always present");
        s.quality.update(quality);
        s.cost.update(cost_usd);
        s.latency.update(latency_ms);

        self.global_cost.update(cost_usd);
        self.global_latency.update(latency_ms);
        self.global_efficiency.update(efficiency);
        self.total_turns += 1;

        RunResult {
            is_correct,
            latency_ms,
            cost_usd,
            efficiency,
            winning_topology: topology,
            budget_mode: mode,
        }
    }

    /// Reset all UCB1 state to simulate a fresh session.
    pub fn reset(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
        for s in self.stats.values_mut() {
            *s = TopologyStats::default();
        }
        self.total_turns = 0;
        self.global_cost = RunningAverage::default();
        self.global_latency = RunningAverage::default();
        self.global_efficiency = RunningAverage::default();
    }

    /// Return the cumulative selection count for every topology.
    pub fn selection_counts(&self) -> HashMap<DebateTopology, u32> {
        self.stats
            .iter()
            .map(|(t, s)| (*t, s.quality.count))
            .collect()
    }

    // ─── Private ──────────────────────────────────────────────────────────────

    /// UCB1 arm selection.
    ///
    /// Any topology with zero observations is selected immediately (exploration
    /// mandate). Once all arms are seeded, the arm with the highest
    /// `efficiency + exploration_bonus` wins.
    fn select_topology(&mut self, mode: BudgetMode) -> DebateTopology {
        // Force-explore unvisited arms first (in fixed order for reproducibility).
        for t in DebateTopology::ALL {
            if self.stats[&t].quality.count == 0 {
                return t;
            }
        }

        let n = self.total_turns as f64;
        let global_eff_mean = self.global_efficiency.mean.max(1e-6);
        let global_cost_mean = self.global_cost.mean.max(1e-6);
        let global_lat_mean = self.global_latency.mean.max(1e-6);

        DebateTopology::ALL
            .iter()
            .copied()
            .map(|t| {
                let s = &self.stats[&t];
                let n_t = s.quality.count as f64;

                let eff = efficiency_formula(
                    s.quality.mean,
                    s.cost.mean,
                    s.latency.mean,
                    global_cost_mean,
                    global_lat_mean,
                    mode.alpha(),
                    mode.beta(),
                );

                // UCB1 exploration bonus scaled by global mean efficiency so the
                // bonus stays dimensionally consistent with the exploitation term.
                let exploration = global_eff_mean * (2.0 * n.ln() / n_t).sqrt();
                (t, eff + exploration)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, _)| t)
            .unwrap_or(DebateTopology::RoundRobin)
    }

    /// Simulate a topology's outcome using per-topology probability distributions.
    ///
    /// Returns `(is_correct, latency_ms, cost_usd)`.
    fn simulate_outcome(&mut self, t: DebateTopology) -> (bool, f64, f64) {
        let is_correct = Bernoulli::new(t.quality_prob())
            .expect("quality_prob is always in [0,1]")
            .sample(&mut self.rng);

        let cost = self.sample_lognormal(t.mean_cost_usd(), t.cost_cv());
        let latency = self.sample_lognormal(t.mean_latency_ms(), t.latency_cv());

        (is_correct, latency, cost)
    }

    /// Compute instantaneous efficiency for result logging.
    ///
    /// When no global baseline exists yet (first call), the single observation
    /// normalizes to 1.0 (norm_cost = norm_latency = 1), which is correct.
    fn compute_efficiency(
        &self,
        quality: f64,
        cost_usd: f64,
        latency_ms: f64,
        mode: BudgetMode,
    ) -> f64 {
        // Bootstrap: use the observation itself as the global mean on the first turn.
        let gc = self.global_cost.mean.max(cost_usd).max(1e-6);
        let gl = self.global_latency.mean.max(latency_ms).max(1e-6);
        efficiency_formula(
            quality,
            cost_usd,
            latency_ms,
            gc,
            gl,
            mode.alpha(),
            mode.beta(),
        )
    }

    /// Draw a log-normal sample with the given mean and coefficient of variation.
    ///
    /// Uses the Box-Muller transform to avoid an additional dependency on rand_distr.
    ///
    /// Parametrisation:
    ///   σ²_ln = ln(CV² + 1)
    ///   μ_ln  = ln(mean) − σ²_ln / 2
    fn sample_lognormal(&mut self, mean: f64, cv: f64) -> f64 {
        let sigma_sq = (cv * cv + 1.0).ln();
        let mu = mean.ln() - sigma_sq / 2.0;
        let sigma = sigma_sq.sqrt();

        // Box-Muller: two uniform draws → one standard normal.
        let u1: f64 = self.rng.random::<f64>().max(f64::EPSILON);
        let u2: f64 = self.rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        (mu + sigma * z).exp()
    }
}

// ─── Efficiency formula (stateless) ───────────────────────────────────────

/// Core efficiency formula from the Crosstalk paper (Section 3.2).
///
/// ```text
/// norm_cost    = (cost   + ε_c) / (global_cost_mean   + ε_c)
/// norm_latency = (latency + ε_l) / (global_latency_mean + ε_l)
/// efficiency   = quality / (α·norm_cost + β·norm_latency + ε_floor)
/// ```
pub fn efficiency_formula(
    quality: f64,
    cost_usd: f64,
    latency_ms: f64,
    global_cost_mean: f64,
    global_lat_mean: f64,
    alpha: f64,
    beta: f64,
) -> f64 {
    const EPS_C: f64 = 1e-4;
    const EPS_L: f64 = 1e-4;
    const EPS_FLOOR: f64 = 0.1;

    let norm_cost = (cost_usd + EPS_C) / (global_cost_mean + EPS_C);
    let norm_latency = (latency_ms + EPS_L) / (global_lat_mean + EPS_L);

    quality / (alpha * norm_cost + beta * norm_latency + EPS_FLOOR)
}

// ─── Result type ──────────────────────────────────────────────────────────

/// The output of a single simulated orchestrator turn.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunResult {
    pub is_correct: bool,
    pub latency_ms: f64,
    pub cost_usd: f64,
    /// Instantaneous efficiency score for this turn.
    pub efficiency: f64,
    /// The topology chosen by UCB1 for this turn.
    pub winning_topology: DebateTopology,
    pub budget_mode: BudgetMode,
}
