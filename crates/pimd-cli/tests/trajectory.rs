//! Trajectory output (MILESTONE_5 "Clarifications: Trajectory output", "Required tests: Trajectory
//! output"; SPEC C-5, U-3, D-3). `trajectory.xyz` holds one frame per production step s with
//! s % trajectory_every == 0 (none during equilibration): line 1 P·N, line 2
//! `step=<s> beads=<P> atoms=<N> units=angstrom`, then P·N lines `symbol x y z`, bead-major,
//! symbol `X` without `symbols`, positions unwrapped (C-5), in Å (U-3), written with `{}`. After a
//! restart it holds only the frames after the checkpoint, so the concatenation equals the
//! uninterrupted file.
//!
//! Expected positions come from `common::reference_run_with` (a plain `RingPolymerSim` loop with
//! the brief's initial state, written from the brief), converted to Å with the U-2 literal
//! 0.529177210903 Å/bohr; compared to 1e-15 relative (a conversion by multiplication or by
//! division by the inverse may differ in the last bit).
//!
//! `trajectory_every` is independent of `stride` (Q19): frames at every production step with
//! s % trajectory_every == 0, also when that step is not a multiple of `stride`.

mod common;

use common::{bytes, lj_params, parse, pool, read_trajectory, reference_run_with, set, text, tmp};
use pimd_cli::{run, run_from_checkpoint};
use pimd_potentials::LennardJones;

const BOHR_A: f64 = 0.529177210903; // Å per bohr (U-2)

/// Four neon atoms (one at the origin, so beads leave [0, L) at once) in a 12.5 Å box, P = 4,
/// LJ as M4's BASE; 23 equilibration steps (not a multiple of trajectory_every, so frames counted
/// from the start of the run would show) and `n_steps` production steps; frames every 10 steps.
fn periodic(n_steps: usize) -> String {
    format!(
        r#"
[system]
n_atoms = 4
mass_amu = 20.1797
box_angstrom = 12.5
positions_angstrom = [[0.0, 0.0, 0.0], [3.1, 0.0, 0.0], [0.0, 3.1, 0.0], [0.0, 0.0, 3.1]]

[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.0
skin_sigma = 0.25
shift = true

[sampling]
temperature_kelvin = 30.0
n_beads = 4
dt_fs = 1.0
tau0_fs = 100.0
n_equil = 23
n_steps = {n_steps}
stride = 5
seed = 777

[output]
checkpoint_every = 10
trajectory_every = 10
"#
    )
}

/// The same atoms in the gas phase, with per-atom masses and symbols.
fn gas(n_steps: usize) -> String {
    let t = periodic(n_steps).replace("box_angstrom = 12.5\n", "");
    let t = set(&t, "potential", "cutoff_sigma", "6.0");
    let t = common::remove(&t, "mass_amu");
    let t = set(
        &t,
        "system",
        "masses_amu",
        "[20.1797, 20.1797, 39.948, 20.1797]",
    );
    set(&t, "system", "symbols", "[\"Ne\", \"Ne\", \"Ar\", \"Ne\"]")
}

/// Compares the trajectory in `dir` with the reference frames of `toml` (steps, symbols, Å).
fn check_against_reference(toml: &str, dir: &std::path::Path, symbols: &[&str]) -> bool {
    let cfg = parse(toml);
    let (p, n) = (cfg.sampling.n_beads, cfg.system.n_atoms);
    let pot = LennardJones::new(lj_params(&cfg)).unwrap();
    let r = reference_run_with(&cfg, Box::new(pot), cfg.output.trajectory_every);
    let got = read_trajectory(&dir.join("trajectory.xyz"), p, n);
    let steps: Vec<usize> = got.iter().map(|f| f.0).collect();
    let want: Vec<usize> = r.frames.iter().map(|f| f.0).collect();
    assert_eq!(steps, want, "frame steps");
    let mut outside = false;
    for ((s, rows), (_, q)) in got.iter().zip(&r.frames) {
        for j in 0..p {
            for i in 0..n {
                let (sym, x) = &rows[j * n + i];
                assert_eq!(sym, symbols[i], "step {s}: symbol of bead {j} atom {i}");
                let want = q.get(j, i);
                for d in 0..3 {
                    let w = want[d] * BOHR_A;
                    let rel = (x[d] - w).abs() / w.abs().max(1e-300);
                    assert!(
                        rel <= 1e-15 || x[d] == w,
                        "step {s}, bead {j}, atom {i}, dim {d}: {} Å vs {w} Å",
                        x[d]
                    );
                    if let Some(c) = cfg.system.cell {
                        let l = c.lengths()[d] * BOHR_A;
                        outside |= x[d] < 0.0 || x[d] >= l;
                    }
                }
            }
        }
    }
    outside
}

/// C-5, U-3 ((new) trajectory output): format, recording rule, unwrapped Å positions, `X` without symbols (periodic).
/// The test checks that some written coordinate lies outside [0, L) (C-5: never wrapped).
#[test]
fn periodic_trajectory_matches_reference() {
    let d = tmp();
    let toml = periodic(40);
    pool(1).install(|| run(&parse(&toml), d.path()).unwrap());
    let outside = check_against_reference(&toml, d.path(), &["X"; 4]);
    assert!(
        outside,
        "test setup: expected unwrapped coordinates outside the box"
    );
    // Frames at 10, 20, 30, 40 (production steps; equilibration writes nothing).
    let frames = read_trajectory(&d.path().join("trajectory.xyz"), 4, 4);
    assert_eq!(
        frames.iter().map(|f| f.0).collect::<Vec<_>>(),
        vec![10, 20, 30, 40]
    );
}

/// C-1, S-2 ((new) trajectory output): a gas-phase run with per-atom masses and symbols.
#[test]
fn gas_phase_trajectory_with_symbols() {
    let d = tmp();
    let toml = gas(30);
    pool(1).install(|| run(&parse(&toml), d.path()).unwrap());
    check_against_reference(&toml, d.path(), &["Ne", "Ne", "Ar", "Ne"]);
}

/// D-3 ((new) trajectory output, restart concatenation): 40 steps = 20 + checkpoint + 20, and 23 + 17 (checkpoint between
/// frames): the restarted trajectory holds only frames after the checkpoint, and the two files
/// concatenated equal the uninterrupted file byte for byte. Also at 4 threads.
#[test]
fn restart_concatenation() {
    for (k, threads) in [(20usize, 1usize), (23, 1), (20, 4)] {
        let (full, a, b) = (tmp(), tmp(), tmp());
        pool(threads).install(|| {
            run(&parse(&periodic(40)), full.path()).unwrap();
            run(&parse(&periodic(k)), a.path()).unwrap();
            run_from_checkpoint(
                &parse(&periodic(40)),
                &a.path().join("checkpoint.bin"),
                b.path(),
            )
            .unwrap();
        });
        for (s, _) in read_trajectory(&b.path().join("trajectory.xyz"), 4, 4) {
            assert!(s > k, "restarted trajectory has frame {s} ≤ {k}");
        }
        let mut joined = bytes(&a.path().join("trajectory.xyz"));
        joined.extend(bytes(&b.path().join("trajectory.xyz")));
        assert_eq!(
            String::from_utf8(joined).unwrap(),
            text(&full.path().join("trajectory.xyz")),
            "k = {k}, {threads} threads"
        );
    }
}

/// Q19 ((new) trajectory output): `trajectory_every` stands alone. With stride = 5 and
/// trajectory_every = 3, frames are written at 3, 6, …, 39 (not only at multiples of 15), and they
/// match the reference positions.
#[test]
fn trajectory_every_independent_of_stride() {
    let d = tmp();
    let toml = set(&periodic(40), "output", "trajectory_every", "3");
    pool(1).install(|| run(&parse(&toml), d.path()).unwrap());
    check_against_reference(&toml, d.path(), &["X"; 4]);
    let frames = read_trajectory(&d.path().join("trajectory.xyz"), 4, 4);
    assert_eq!(
        frames.iter().map(|f| f.0).collect::<Vec<_>>(),
        (3..=39).step_by(3).collect::<Vec<_>>()
    );
}
