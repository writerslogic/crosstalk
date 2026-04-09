use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use crosstalk::engines::consensus::{CertaintyAnalyzer, NashSolver, ResolutionStrategy};
use crosstalk::engines::diff::DiffEngine;
use crosstalk::core::state::StateManager;
use crosstalk::types::conversation::ConversationState;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Diff computation
// ---------------------------------------------------------------------------

fn bench_diff(c: &mut Criterion) {
    let small = ("fn foo() {}\n".repeat(10), "fn foo() { 1 }\n".repeat(10));
    let large = ("fn foo() {}\n".repeat(500), "fn foo() { 1 }\n".repeat(500));

    let mut group = c.benchmark_group("diff/generate_delta");
    for (label, (old, new)) in [("small_10l", &small), ("large_500l", &large)] {
        group.bench_with_input(BenchmarkId::new("lines", label), &(old, new), |b, (o, n)| {
            b.iter(|| DiffEngine::generate_delta(black_box(o), black_box(n), 0));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Consensus resolution
// ---------------------------------------------------------------------------

fn bench_consensus(c: &mut Criterion) {
    let proposals_2: Vec<(&str, f64, &str)> =
        vec![("a", 0.6, "use async"), ("b", 0.4, "use sync")];
    let proposals_8: Vec<(&str, f64, &str)> = (0..8)
        .map(|i| {
            let weight = 1.0 / 8.0;
            // leak is acceptable in bench harness; strings are tiny and long-lived
            let text: &'static str = Box::leak(format!("proposal_{i}").into_boxed_str());
            let id: &'static str = Box::leak(format!("agent_{i}").into_boxed_str());
            (id, weight, text)
        })
        .collect();

    let mut group = c.benchmark_group("consensus/resolve");
    for (label, proposals) in [("2_agents", &proposals_2), ("8_agents", &proposals_8)] {
        group.bench_with_input(
            BenchmarkId::new("weighted_average", label),
            proposals,
            |b, p| {
                b.iter(|| NashSolver::resolve(black_box(p), ResolutionStrategy::WeightedAverage));
            },
        );
        group.bench_with_input(BenchmarkId::new("voting", label), proposals, |b, p| {
            b.iter(|| NashSolver::resolve(black_box(p), ResolutionStrategy::Voting));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Certainty analysis
// ---------------------------------------------------------------------------

fn bench_certainty(c: &mut Criterion) {
    let hedged = "maybe this could possibly be the right approach, i think";
    let confident = "this is definitely correct and must be the optimal fix";

    let mut group = c.benchmark_group("consensus/certainty");
    group.bench_function("hedged", |b| {
        b.iter(|| CertaintyAnalyzer::compute(black_box(hedged), black_box(0.3)));
    });
    group.bench_function("confident", |b| {
        b.iter(|| CertaintyAnalyzer::compute(black_box(confident), black_box(0.1)));
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// State transitions (Sled checkpoint + restore)
// ---------------------------------------------------------------------------

fn bench_state(c: &mut Criterion) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_str().expect("utf8 path").to_string();
    let mgr = StateManager::new(&path).expect("state manager");

    let mut state = ConversationState::new("bench-session");
    // Warm up: write an initial checkpoint so restore has something to find.
    mgr.checkpoint(&state).expect("initial checkpoint");

    let mut group = c.benchmark_group("state");
    group.bench_function("checkpoint", |b| {
        b.iter(|| {
            state.iteration_index += 1;
            mgr.checkpoint(black_box(&state)).expect("checkpoint");
        });
    });

    // Restore the first written index.
    group.bench_function("restore", |b| {
        b.iter(|| {
            mgr.restore(black_box(0)).expect("restore");
        });
    });
    group.finish();

    // Keep dir alive until group is finished.
    drop(dir);
}

criterion_group!(benches, bench_diff, bench_consensus, bench_certainty, bench_state);
criterion_main!(benches);
