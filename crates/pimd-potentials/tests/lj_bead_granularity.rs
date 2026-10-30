//! D-3, P-3, N-3 (MILESTONE_4 "LJ granularity"): parallelising `LennardJones::evaluate` over all
//! (bead, atom) pairs must leave results bitwise unchanged. A P-bead evaluation equals P separate
//! single-bead evaluations (each with its own fresh `LennardJones`) bit for bit, energies and
//! forces, inside 1-thread and 4-thread pools; also after the per-bead lists were rebuilt, and
//! when beads need rebuilds at different steps.
//!
//! The reference is the single-bead path (P = 1), whose per-atom terms and index-order reduction
//! M3 fixed; a (bead, atom) split that changes either (e.g. a reduction over beads × atoms in one
//! rayon `sum`) breaks bit equality.

mod common;

use common::{SplitMix64, bead, beads, beads1, random_config, unwrap_randomly};
use pimd_core::{Beads, Cell};
use pimd_potentials::{LennardJones, LjParams, Potential};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

fn params(shift: bool) -> LjParams {
    LjParams {
        epsilon: 0.7,
        sigma: 1.0,
        cutoff: 2.5,
        skin: 0.4,
        shift,
    }
}

/// P configurations of `n` atoms in `l`, unwrapped by random multiples of L (C-5).
fn configs(seed: u64, p: usize, n: usize, l: [f64; 3]) -> Vec<Vec<[f64; 3]>> {
    let mut rng = SplitMix64(seed);
    (0..p)
        .map(|_| {
            let x = random_config(&mut rng, n, l, 0.85, true);
            unwrap_randomly(&mut rng, &x, l)
        })
        .collect()
}

/// Asserts `all` (a P-bead evaluation) equals the per-bead `single` evaluations bitwise.
fn assert_matches_single(
    all: &pimd_potentials::Evaluation,
    single: &[pimd_potentials::Evaluation],
    what: &str,
) {
    let p = single.len();
    assert_eq!(all.energies.len(), p);
    let n = all.forces.n_atoms();
    for (j, s) in single.iter().enumerate() {
        assert_eq!(
            all.energies[j].to_bits(),
            s.energies[0].to_bits(),
            "{what}: energy of bead {j}: {} vs {}",
            all.energies[j],
            s.energies[0]
        );
        for i in 0..n {
            let (a, b) = (all.forces.get(j, i), s.forces.get(0, i));
            for d in 0..3 {
                assert_eq!(
                    a[d].to_bits(),
                    b[d].to_bits(),
                    "{what}: force (bead {j}, atom {i}, {d}): {} vs {}",
                    a[d],
                    b[d]
                );
            }
        }
    }
}

/// Evaluates every bead of `q` separately with its own fresh `LennardJones`.
fn single_bead(q: &Beads, cell: &Cell, prm: LjParams) -> Vec<pimd_potentials::Evaluation> {
    (0..q.n_beads())
        .map(|j| {
            let mut pot = LennardJones::new(prm).unwrap();
            pot.evaluate(&beads1(&bead(q, j)), Some(cell), None)
                .unwrap()
        })
        .collect()
}

/// D-3, P-3: N = 250, P ∈ {2, 5, 16}, shifted and unshifted, a non-cubic cell: the P-bead
/// evaluation equals P single-bead evaluations bitwise, at 1 and 4 threads.
#[test]
fn p_bead_equals_single_beads_bitwise() {
    let l = [10.5, 11.0, 12.0];
    let cell = Cell::new(l).unwrap();
    for (p, seed) in [(2usize, 1u64), (5, 2), (16, 3)] {
        let q = beads(&configs(0xB0 + seed, p, 250, l));
        for shift in [true, false] {
            let prm = params(shift);
            let reference = pool(1).install(|| single_bead(&q, &cell, prm));
            for threads in [1usize, 4] {
                let all = pool(threads).install(|| {
                    let mut pot = LennardJones::new(prm).unwrap();
                    pot.evaluate(&q, Some(&cell), None).unwrap()
                });
                assert_matches_single(
                    &all,
                    &reference,
                    &format!("P = {p}, shift {shift}, {threads} threads"),
                );
            }
        }
    }
}

/// D-3, N-2, N-3: with one long-lived object, a sequence of evaluations in which only some beads
/// move by more than skin/2 (so the beads' lists are rebuilt at different calls) still equals
/// fresh single-bead evaluations bitwise after every call, at 1 and 4 threads.
#[test]
fn staggered_rebuilds_stay_bitwise() {
    let l = [11.0, 11.0, 11.0];
    let cell = Cell::new(l).unwrap();
    let p = 6;
    let mut conf = configs(0xC0, p, 200, l);
    let prm = params(true);
    let mut seqs = Vec::new();
    for round in 0..4 {
        // Beads j with (j + round) % 3 == 0 move by 0.25 (> skin/2 = 0.2) along a diagonal; the
        // others by 0.05 (no rebuild).
        for (j, c) in conf.iter_mut().enumerate() {
            let s = if (j + round) % 3 == 0 { 0.25 } else { 0.05 };
            for r in c.iter_mut() {
                r[0] += s;
                r[1] -= 0.5 * s;
            }
        }
        seqs.push(beads(&conf));
    }
    let reference: Vec<_> = seqs.iter().map(|q| single_bead(q, &cell, prm)).collect();
    for threads in [1usize, 4] {
        let results = pool(threads).install(|| {
            let mut pot = LennardJones::new(prm).unwrap();
            seqs.iter()
                .map(|q| pot.evaluate(q, Some(&cell), None).unwrap())
                .collect::<Vec<_>>()
        });
        for (k, (all, single)) in results.iter().zip(&reference).enumerate() {
            assert_matches_single(all, single, &format!("call {k}, {threads} threads"));
        }
    }
}
