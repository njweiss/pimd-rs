//! The run driver against a loop written in the test (MILESTONE_4 "Required tests: P = 1 limit";
//! SPEC RP-6, RP-3, I-1, T-1, E-1 … E-4, E-7, D-2). `common::reference_run` builds the pinned
//! initial lattice and momenta, runs `RingPolymerSim` (PILE-L, `LennardJones`, same seed) and
//! records with the M2/M3 estimators; every recorded value of observables.csv must equal it to
//! 1e-12 relative, as must the summary means and the RDF.
//!
//! P = 1 is the brief's RP-6 check. The same comparison at P = 4 (not required by the brief, added
//! so that per-bead vs per-atom averaging, the momentum variance m·P/β, the bead initialisation
//! and the P-dependent estimator terms are pinned too) uses 1e-10: a driver that does not reuse
//! `RingPolymerSim` may differ by summation order; such 1e-16 differences grow only slowly in this
//! dilute 32-atom system (the fft/matrix runs, which differ in exactly this way, stay within 1e-13
//! relative over 2000 steps with the reference implementation). The reference implementation,
//! which reuses `RingPolymerSim`, matches bitwise.

mod common;

use common::{BASE, parse, pool, reference_run, rel, set, tmp};
use pimd_cli::{Config, run};

fn compare(cfg: &Config, tol: f64, what: &str) {
    let d = tmp();
    let s = pool(1).install(|| run(cfg, d.path()).unwrap());
    let want = reference_run(cfg);
    let got = common::observables(d.path());
    assert_eq!(got.len(), want.rows.len(), "{what}: number of rows");
    assert!(!want.rows.is_empty());
    for (g, w) in got.iter().zip(&want.rows) {
        assert_eq!(g[0], w.step as f64, "{what}: step");
        for (c, name, v) in [
            (1, "e_prim_eh", w.e_prim),
            (2, "e_vir_eh", w.e_vir),
            (3, "v_eh", w.v),
            (4, "t_est_k", w.t_est),
        ] {
            let r = rel(g[c], v, 1e-300);
            assert!(
                r <= tol,
                "{what}: step {} {name}: {} vs reference {v} ({r:e})",
                w.step,
                g[c]
            );
        }
    }
    let n = want.rows.len() as f64;
    let mp = want.rows.iter().map(|r| r.e_prim).sum::<f64>() / n;
    let mv = want.rows.iter().map(|r| r.e_vir).sum::<f64>() / n;
    assert!(
        rel(s.mean_e_prim, mp, 1e-300) <= tol,
        "{what}: mean_e_prim {} vs {mp}",
        s.mean_e_prim
    );
    assert!(
        rel(s.mean_e_vir, mv, 1e-300) <= tol,
        "{what}: mean_e_vir {} vs {mv}",
        s.mean_e_vir
    );
    let (rg, gg) = s.rdf.expect("rdf");
    let (rw, gw) = want.rdf.unwrap();
    assert_eq!(rg.len(), rw.len());
    for b in 0..rg.len() {
        assert!(
            rel(rg[b], rw[b], 1e-300) <= 1e-14,
            "{what}: rdf bin centre {b}"
        );
        // A pair crossing a bin edge between the trajectories would move a whole count; at these
        // tolerances it does not happen (checked), so the histograms must agree to rounding.
        assert!(
            (gg[b] - gw[b]).abs() <= tol * gw[b].abs().max(1.0),
            "{what}: g[{b}] {} vs {}",
            gg[b],
            gw[b]
        );
    }
}

/// RP-6, I-1, T-1, E-4, E-7: n_beads = 1 (classical limit), N = 32, 60 production steps after
/// 20 equilibration steps, stride 3: the recorded observables equal the RingPolymerSim loop to
/// 1e-12 relative per value; also the summary means and g(r).
#[test]
fn p1_matches_ring_polymer_sim_loop() {
    let toml = set(
        &set(
            &set(BASE, "sampling", "n_beads", "1"),
            "sampling",
            "n_steps",
            "60",
        ),
        "sampling",
        "stride",
        "3",
    );
    compare(&parse(&toml), 1e-12, "P = 1");
}

/// RP-6 with a perfect-cube atom count: N = 27 must use n_side = ⌈27^(1/3)⌉ = 3 exactly (an
/// idiom such as ⌊N^(1/3)⌋ + 1 gives 4, and floating cube roots of cubes are not exact on every
/// platform: glibc's `cbrt(27.0)` is 3.0000000000000004). Box 11.5 Å, cutoff 1.8σ + skin 0.25σ =
/// 5.64 Å ≤ 5.75 Å.
#[test]
fn p1_perfect_cube_lattice() {
    let mut toml = set(BASE, "sampling", "n_beads", "1");
    for (s, k, v) in [
        ("system", "n_atoms", "27"),
        ("system", "box_angstrom", "11.5"),
        ("potential", "cutoff_sigma", "1.8"),
        ("output", "rdf_rmax_angstrom", "5.0"),
        ("sampling", "n_steps", "30"),
    ] {
        toml = set(&toml, s, k, v);
    }
    compare(&parse(&toml), 1e-12, "P = 1, N = 27");
}

/// RP-3, E-1 … E-4, E-7: the same comparison at P = 4 (per-atom averages, momentum variance
/// m·P/β, all beads at the lattice site, atom-major draw order), 60 production steps, 1e-10.
#[test]
fn p4_matches_ring_polymer_sim_loop() {
    let toml = set(BASE, "sampling", "n_steps", "60");
    compare(&parse(&toml), 1e-10, "P = 4");
}

/// RP-3, E-7: P = 3 (odd) with a non-cubic box, no equilibration, stride 1.
#[test]
fn p3_noncubic_matches_loop() {
    let mut toml = set(BASE, "sampling", "n_beads", "3");
    for (s, k, v) in [
        ("system", "box_angstrom", "[12.5, 13.5, 14.5]"),
        ("sampling", "n_equil", "0"),
        ("sampling", "stride", "1"),
        ("sampling", "n_steps", "30"),
    ] {
        toml = set(&toml, s, k, v);
    }
    compare(&parse(&toml), 1e-10, "P = 3, non-cubic");
}
