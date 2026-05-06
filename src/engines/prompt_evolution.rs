//! DSPy-inspired recursive prompt evolution via tournament selection.
//!
//! Maintains a population of [`PromptTemplate`]s and advances them through
//! successive generations using tournament selection, single-point crossover,
//! and stochastic mutation.  Elo ratings from a
//! [`MetacognitiveObserver`](crate::engines::metacognition::MetacognitiveObserver)
//! can bias template assignment toward higher-performing agents.

use crate::types::intelligence::{MutationStrategy, PromptTemplate};
use rand::Rng;
use rustc_hash::FxHashMap;

/// Default number of candidates sampled in each tournament round.
const DEFAULT_TOURNAMENT_SIZE: usize = 3;
/// Default per-offspring mutation probability.
const DEFAULT_MUTATION_RATE: f64 = 0.3;
/// Default number of elite templates preserved unchanged per generation.
const DEFAULT_ELITE_COUNT: usize = 2;
/// Default population cap.
const DEFAULT_MAX_POPULATION: usize = 20;

/// Minimum number of recorded observations before a template is eligible
/// for culling.
const CULL_MIN_OBSERVATIONS: usize = 5;

/// Reasoning directives appended during [`MutationStrategy::Append`] mutation.
static REASONING_DIRECTIVES: &[&str] = &[
    "Think step by step.",
    "Consider counterarguments before concluding.",
    "Identify the key assumptions underlying your answer.",
    "Explain your reasoning before giving the final answer.",
    "Check your answer for logical consistency.",
];

/// Role-framing prefixes applied during [`MutationStrategy::Prefix`] mutation.
static ROLE_FRAMINGS: &[&str] = &[
    "You are a rigorous analyst who values precision over brevity.",
    "You are a creative synthesizer who connects disparate ideas.",
    "You are a skeptical reviewer who challenges every assumption.",
    "You are a domain expert who cites evidence for every claim.",
    "You are a systematic engineer who solves problems step by step.",
];

/// Manages a population of [`PromptTemplate`]s and evolves them through
/// tournament-selection-based genetic iteration.
pub struct PromptEvolver {
    /// Current generation of prompt templates.
    pub population: Vec<PromptTemplate>,
    /// Current generation index (starts at 0, incremented after each [`evolve`](Self::evolve)).
    pub generation: u32,
    /// Number of candidates sampled per tournament round.
    pub tournament_size: usize,
    /// Probability that an offspring is mutated after crossover.
    pub mutation_rate: f64,
    /// Number of top-performing templates preserved unchanged each generation.
    pub elite_count: usize,
    /// Maximum number of templates retained in the population.
    pub max_population: usize,
}

impl Default for PromptEvolver {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptEvolver {
    /// Create a [`PromptEvolver`] with default hyperparameters and an empty population.
    pub fn new() -> Self {
        Self {
            population: Vec::new(),
            generation: 0,
            tournament_size: DEFAULT_TOURNAMENT_SIZE,
            mutation_rate: DEFAULT_MUTATION_RATE,
            elite_count: DEFAULT_ELITE_COUNT,
            max_population: DEFAULT_MAX_POPULATION,
        }
    }

    /// Seed the evolver with an initial population.
    pub fn seed(&mut self, templates: Vec<PromptTemplate>) {
        self.population = templates;
        self.population.truncate(self.max_population);
    }

    /// Tournament selection: sample `tournament_size` templates uniformly at
    /// random and return the one with the highest [`mean_performance`](PromptTemplate::mean_performance).
    ///
    /// Returns `None` if the population is empty.
    pub fn select_parent(&self) -> Option<&PromptTemplate> {
        if self.population.is_empty() {
            return None;
        }
        let mut rng = rand::rng();
        let k = self.tournament_size.min(self.population.len());
        let mut best_idx = rng.random_range(0..self.population.len());
        for _ in 1..k {
            let idx = rng.random_range(0..self.population.len());
            if self.population[idx].mean_performance()
                > self.population[best_idx].mean_performance()
            {
                best_idx = idx;
            }
        }
        Some(&self.population[best_idx])
    }

    /// Single-point crossover: the first half of `parent_a`'s template text is
    /// concatenated with the second half of `parent_b`'s. Variables are merged
    /// (deduplicated).  The resulting template inherits `parent_a`'s id with a
    /// `_x` suffix, version 0, and an empty performance history.
    pub fn crossover(parent_a: &PromptTemplate, parent_b: &PromptTemplate) -> PromptTemplate {
        let a_chars: Vec<char> = parent_a.template_text.chars().collect();
        let b_chars: Vec<char> = parent_b.template_text.chars().collect();

        let cut_a = a_chars.len() / 2;
        let cut_b = b_chars.len() / 2;

        let first_half: String = a_chars[..cut_a].iter().collect();
        let second_half: String = b_chars[cut_b..].iter().collect();
        let text = format!("{}{}", first_half, second_half);

        let mut variables = parent_a.variables.clone();
        for v in &parent_b.variables {
            if !variables.contains(v) {
                variables.push(v.clone());
            }
        }

        PromptTemplate {
            id: format!("{}_x", parent_a.id),
            version: 0,
            template_text: text,
            task_category: parent_a.task_category,
            variables,
            tags: parent_a.tags.clone(),
            performance_history: Vec::new(),
        }
    }

    /// Apply a randomly chosen [`MutationStrategy`] to produce a mutated copy:
    ///
    /// - 25%: Append a reasoning directive
    /// - 25%: Prefix a role framing
    /// - 25%: [`InjectExamples`](MutationStrategy::InjectExamples)
    /// - 25%: Trim to 80% of current character length
    pub fn mutate_template(template: &PromptTemplate) -> PromptTemplate {
        let mut rng = rand::rng();
        let roll: u32 = rng.random_range(0..4);
        let strategy = match roll {
            0 => {
                let idx = rng.random_range(0..REASONING_DIRECTIVES.len());
                MutationStrategy::Append(REASONING_DIRECTIVES[idx].to_string())
            }
            1 => {
                let idx = rng.random_range(0..ROLE_FRAMINGS.len());
                MutationStrategy::Prefix(ROLE_FRAMINGS[idx].to_string())
            }
            2 => MutationStrategy::InjectExamples,
            _ => {
                let len = template.template_text.chars().count();
                let trimmed = (len * 4 / 5).max(1);
                MutationStrategy::Trim(trimmed)
            }
        };
        template.mutate(strategy)
    }

    /// Advance the population by one generation.
    ///
    /// 1. Sort current population by descending [`mean_performance`](PromptTemplate::mean_performance).
    /// 2. Preserve the top `elite_count` templates unchanged.
    /// 3. Fill remaining slots via tournament selection + crossover, then
    ///    apply mutation with probability [`mutation_rate`](Self::mutation_rate).
    /// 4. Clamp to [`max_population`](Self::max_population).
    /// 5. Increment [`generation`](Self::generation).
    ///
    /// Returns the new population (which also replaces `self.population`).
    pub fn evolve(&mut self) -> Vec<PromptTemplate> {
        if self.population.is_empty() {
            return Vec::new();
        }

        self.population.sort_by(|a, b| {
            b.mean_performance()
                .partial_cmp(&a.mean_performance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let elite_n = self.elite_count.min(self.population.len());
        let mut next_gen: Vec<PromptTemplate> = self.population[..elite_n].to_vec();

        let target = self.max_population.min(self.population.len().max(elite_n));
        let mut rng = rand::rng();

        while next_gen.len() < target {
            let parent_a = match self.select_parent() {
                Some(p) => p.clone(),
                None => break,
            };
            let parent_b = match self.select_parent() {
                Some(p) => p.clone(),
                None => break,
            };
            let mut offspring = Self::crossover(&parent_a, &parent_b);
            if rng.random::<f64>() < self.mutation_rate {
                offspring = Self::mutate_template(&offspring);
            }
            next_gen.push(offspring);
        }

        next_gen.truncate(self.max_population);
        self.generation += 1;
        self.population = next_gen.clone();
        next_gen
    }

    /// Select the best template for `agent_id`, weighting template
    /// [`mean_performance`](PromptTemplate::mean_performance) by the agent's
    /// normalized Elo score.
    ///
    /// The combined score for each template is:
    /// `template_perf * (1 + elo_norm)` where `elo_norm` maps the agent's Elo
    /// to `[0, 1]` relative to the field range.  If the agent has no recorded
    /// Elo, the raw template performance is used.
    ///
    /// Returns `None` if the population is empty.
    pub fn select_for_agent<'a>(
        &'a self,
        agent_id: &str,
        elo_ratings: &FxHashMap<String, f64>,
    ) -> Option<&'a PromptTemplate> {
        if self.population.is_empty() {
            return None;
        }

        let elo_norm = if elo_ratings.is_empty() {
            0.5
        } else {
            let agent_elo = elo_ratings.get(agent_id).copied().unwrap_or(1500.0);
            let min_elo = elo_ratings.values().cloned().fold(f64::INFINITY, f64::min);
            let max_elo = elo_ratings
                .values()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            if (max_elo - min_elo).abs() < f64::EPSILON {
                0.5
            } else {
                (agent_elo - min_elo) / (max_elo - min_elo)
            }
        };

        self.population.iter().max_by(|a, b| {
            let score_a = a.mean_performance() * (1.0 + elo_norm);
            let score_b = b.mean_performance() * (1.0 + elo_norm);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Record a quality outcome for the template identified by `template_id`.
    ///
    /// Delegates to [`PromptTemplate::record_performance`].  Does nothing if
    /// no template with the given id is found.
    pub fn record_outcome(&mut self, template_id: &str, quality: f64) {
        if let Some(t) = self.population.iter_mut().find(|t| t.id == template_id) {
            t.record_performance(template_id.to_string(), quality);
        }
    }

    /// Remove templates whose [`mean_performance`](PromptTemplate::mean_performance)
    /// falls below `min_quality`, but only after they have accumulated at least
    /// [`CULL_MIN_OBSERVATIONS`] observations.  Elite templates (the current
    /// top `elite_count`) are never culled.
    pub fn cull_underperformers(&mut self, min_quality: f64) {
        if self.population.len() <= self.elite_count {
            return;
        }
        self.population.sort_by(|a, b| {
            b.mean_performance()
                .partial_cmp(&a.mean_performance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let elite_n = self.elite_count.min(self.population.len());
        let (elite, rest) = self.population.split_at(elite_n);

        let survivors: Vec<PromptTemplate> = rest
            .iter()
            .filter(|t| {
                t.performance_history.len() < CULL_MIN_OBSERVATIONS
                    || t.mean_performance() >= min_quality
            })
            .cloned()
            .collect();

        self.population = elite.iter().cloned().chain(survivors).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::conversation::TaskCategory;

    fn make_template(id: &str, text: &str, perf: f64) -> PromptTemplate {
        let mut t = PromptTemplate {
            id: id.to_string(),
            version: 0,
            template_text: text.to_string(),
            task_category: TaskCategory::CodeGeneration,
            variables: vec![],
            performance_history: vec![],
        };
        for i in 0..6 {
            t.record_performance(format!("obs_{i}"), perf);
        }
        t
    }

    #[test]
    fn tournament_selects_best() {
        let mut evolver = PromptEvolver::new();
        evolver.population = vec![
            make_template("low", "low quality prompt", 0.2),
            make_template("high", "high quality prompt", 0.9),
            make_template("mid", "medium quality prompt", 0.5),
        ];
        // With 3 candidates the tournament always sees all three, so high wins.
        let winner = evolver.select_parent().unwrap();
        assert_eq!(winner.id, "high");
    }

    #[test]
    fn crossover_produces_hybrid_text() {
        let a = make_template("a", "AAAAAABBBB", 0.7);
        let b = make_template("b", "CCCCCCDDDD", 0.6);
        let child = PromptEvolver::crossover(&a, &b);
        assert!(child.template_text.starts_with("AAAAA"));
        assert!(child.template_text.ends_with("DDDD"));
    }

    #[test]
    fn evolve_increments_generation() {
        let mut evolver = PromptEvolver::new();
        evolver.seed(vec![
            make_template("t1", "first template text here", 0.8),
            make_template("t2", "second template text here", 0.4),
            make_template("t3", "third template text here", 0.6),
        ]);
        evolver.evolve();
        assert_eq!(evolver.generation, 1);
    }

    #[test]
    fn cull_removes_low_performers() {
        let mut evolver = PromptEvolver::new();
        evolver.elite_count = 1;
        evolver.population = vec![
            make_template("top", "best prompt text here", 0.9),
            make_template("bad", "worst prompt text here", 0.1),
        ];
        evolver.cull_underperformers(0.5);
        assert!(!evolver.population.iter().any(|t| t.id == "bad"));
    }

    #[test]
    fn record_outcome_updates_history() {
        let mut evolver = PromptEvolver::new();
        evolver.population = vec![make_template("t", "some template", 0.5)];
        evolver.record_outcome("t", 0.95);
        assert_eq!(evolver.population[0].performance_history.len(), 7);
    }

    #[test]
    fn select_for_agent_returns_template() {
        let mut evolver = PromptEvolver::new();
        evolver.population = vec![make_template("t", "template text", 0.7)];
        let ratings = FxHashMap::default();
        assert!(evolver.select_for_agent("agent_1", &ratings).is_some());
    }
}

pub struct ClosedLoopFeedback;

impl ClosedLoopFeedback {
    #[must_use]
    pub fn generate_corrective_directive(profile: &crate::types::intelligence::AgentProfile) -> Option<String> {
        if profile.total_turns > 3 && profile.compilation_success_rate < 0.6 {
            Some("CRITICAL: Your recent proposals have consistently failed to compile. You MUST verify structural syntax and type signatures before responding.".to_string())
        } else if profile.total_turns > 5 && profile.compilation_success_rate < 0.4 {
             Some("WARNING: High failure rate detected. Switch to a conservative implementation strategy and avoid complex language features.".to_string())
        } else {
            None
        }
    }
}
