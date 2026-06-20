//! CSV report generation for arXiv publication figures.
//!
//! # Output files
//!
//! | File                        | Figure                                  |
//! |-----------------------------|-----------------------------------------|
//! | `topology_distribution.csv` | "Topology Distribution vs. Budget Mode" |
//! | `ucb1_convergence.csv`      | "Cost-Efficiency over Time"             |

use anyhow::Result;
use csv::Writer;
use std::path::Path;

use crate::{
    harness::DebateTopology,
    scenarios::{BudgetPressureRecord, ConvergenceRecord},
};

// ─── Figure 1: Topology Distribution vs. Budget Mode ─────────────────────────

/// Write Scenario 1 records to CSV.
///
/// Schema: `run_index, phase, topology, is_correct, latency_ms, cost_usd, efficiency`
pub fn write_budget_pressure_csv(path: &Path, records: &[BudgetPressureRecord]) -> Result<()> {
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record([
        "run_index",
        "phase",
        "topology",
        "is_correct",
        "latency_ms",
        "cost_usd",
        "efficiency",
    ])?;
    for r in records {
        wtr.write_record([
            r.run_index.to_string(),
            r.phase.to_string(),
            r.result.winning_topology.name().to_string(),
            (r.result.is_correct as u8).to_string(),
            format!("{:.2}", r.result.latency_ms),
            format!("{:.6}", r.result.cost_usd),
            format!("{:.6}", r.result.efficiency),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

/// Print a summary table of topology counts by phase to stdout.
pub fn print_budget_pressure_summary(records: &[BudgetPressureRecord]) {
    use std::collections::HashMap;

    let mut counts: HashMap<(&str, &str), usize> = HashMap::new();
    for r in records {
        *counts
            .entry((r.phase, r.result.winning_topology.name()))
            .or_insert(0) += 1;
    }

    println!("\n=== Scenario 1: Topology Distribution vs. Budget Mode ===");
    println!("{:<20} {:>10} {:>12}", "Topology", "Normal", "Emergency");
    println!("{}", "-".repeat(44));
    for t in DebateTopology::ALL {
        let n = counts.get(&("normal", t.name())).copied().unwrap_or(0);
        let e = counts.get(&("emergency", t.name())).copied().unwrap_or(0);
        println!("{:<20} {:>10} {:>12}", t.name(), n, e);
    }
    println!();
}

// ─── Figure 2: Cost-Efficiency over Time ─────────────────────────────────────

/// Write Scenario 2 records to CSV.
///
/// Schema:
/// `run_index, topology, is_correct, latency_ms, cost_usd, efficiency,`
/// `cumulative_quality_mean,`
/// `count_RoundRobin, count_Adversarial, count_Ensemble,`
/// `count_TreeOfThoughts, count_Mediated, count_Critique`
pub fn write_ucb1_convergence_csv(path: &Path, records: &[ConvergenceRecord]) -> Result<()> {
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record([
        "run_index",
        "topology",
        "is_correct",
        "latency_ms",
        "cost_usd",
        "efficiency",
        "cumulative_quality_mean",
        "count_RoundRobin",
        "count_Adversarial",
        "count_Ensemble",
        "count_TreeOfThoughts",
        "count_Mediated",
        "count_Critique",
    ])?;
    for r in records {
        wtr.write_record([
            r.run_index.to_string(),
            r.result.winning_topology.name().to_string(),
            (r.result.is_correct as u8).to_string(),
            format!("{:.2}", r.result.latency_ms),
            format!("{:.6}", r.result.cost_usd),
            format!("{:.6}", r.result.efficiency),
            format!("{:.6}", r.cumulative_quality_mean),
            r.count_round_robin.to_string(),
            r.count_adversarial.to_string(),
            r.count_ensemble.to_string(),
            r.count_tree_of_thoughts.to_string(),
            r.count_mediated.to_string(),
            r.count_critique.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

/// Print cumulative selection counts at N = 50, 100, 200.
pub fn print_ucb1_convergence_summary(records: &[ConvergenceRecord]) {
    println!("\n=== Scenario 2: UCB1 Convergence (cumulative selection counts) ===");
    println!(
        "{:<20} {:>8} {:>8} {:>8}",
        "Topology", "N=50", "N=100", "N=200"
    );
    println!("{}", "-".repeat(46));

    macro_rules! row {
        ($label:expr, $field:ident) => {
            let c50 = records.get(49).map(|r| r.$field).unwrap_or(0);
            let c100 = records.get(99).map(|r| r.$field).unwrap_or(0);
            let c200 = records.get(199).map(|r| r.$field).unwrap_or(0);
            println!("{:<20} {:>8} {:>8} {:>8}", $label, c50, c100, c200);
        };
    }

    row!("RoundRobin", count_round_robin);
    row!("Adversarial", count_adversarial);
    row!("Ensemble", count_ensemble);
    row!("TreeOfThoughts", count_tree_of_thoughts);
    row!("Mediated", count_mediated);
    row!("Critique", count_critique);

    if let Some(last) = records.last() {
        println!(
            "\nFinal cumulative quality mean: {:.3}",
            last.cumulative_quality_mean
        );
    }
    println!();
}
