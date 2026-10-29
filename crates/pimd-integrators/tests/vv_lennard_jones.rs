//! `VelocityVerlet` with `LennardJones` on the 256-atom LJ liquid (ρ* = 0.8, T* = 1, r_c = 2.5σ,
//! shifted): LAMMPS `fix nve` trajectory (SPEC I-4, P-3), energy conservation with dt² scaling,
//! and bitwise determinism across thread counts (D-3).
//!
//! The snapshot and the LAMMPS trajectory come from
//! `crates/pimd-potentials/tests/data/lj_snapshot.json` (generator `gen_lj_snapshot.py` next to it;
//! LAMMPS version and inputs recorded in the file). LAMMPS lj units = engine atomic units with
//! ε = σ = m = 1, so dt = 0.005 τ is 0.005 ħ/Eh and momenta equal velocities.

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

use lammps::{Snapshot, snapshot_json};
use pimd_core::{Beads, Cell};
use pimd_integrators::{RingState, VelocityVerlet};
use pimd_potentials::{LennardJones, LjParams, Potential};

fn lj(s: &Snapshot) -> LennardJones {
    LennardJones::new(LjParams {
        epsilon: s.epsilon,
        sigma: s.sigma,
        cutoff: s.cutoff,
        skin: s.skin,
        shift: true,
    })
    .unwrap()
}

/// Initial state, forces (the one initialising evaluation, I-3) and potential.
fn setup(s: &Snapshot) -> (RingState, Beads, LennardJones, Cell, f64) {
    let cell = s.cell();
    let mut pot = lj(s);
    let ev = pot.evaluate(&s.q, Some(&cell), None).unwrap();
    let state = RingState {
        q: s.q.clone(),
        p: s.p.clone(),
        masses: s.masses(),
    };
    (state, ev.forces, pot, cell, ev.energies[0])
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn kinetic(state: &RingState) -> f64 {
    let p = state.p.as_slice();
    (0..state.masses.len())
        .map(|i| (0..3).map(|d| p[3 * i + d].powi(2)).sum::<f64>() / (2.0 * state.masses[i]))
        .sum()
}

/// I-4, P-3, C-5: 20 steps of NVE (dt = 0.005) from the snapshot match LAMMPS `fix nve`
/// (unwrapped `xu` positions) after steps 1, 2, 5, 10 and 20: positions and velocities to 1e-8
/// absolute (MILESTONE_3), potential energy after every step to 1e-10 relative.
///
/// Both codes start from the same doubles and run the same algorithm in f64; they differ only in
/// force summation order (≈1e-14 relative per force, see `lj_lammps.rs`). Measured with the
/// reference implementation: max |Δx| = 9e-16 and max |Δv| = 3e-14 after 20 steps. The separation
/// grows about tenfold every 100 steps (Lyapunov instability of the liquid, λ ≈ 4.6 per LJ time
/// unit), so agreement degrades beyond this horizon. Reference vs a 6 000-step LAMMPS run from the
/// same input, max |Δx|: 8e-14 (200 steps), 4e-11 (500), 2e-8 (800), 1.5e-6 (1 000), 0.14 (1 500),
/// 2.5 (2 000, i.e. uncorrelated). Twenty steps sit far inside the 1e-8 band.
#[test]
fn trajectory_matches_lammps() {
    let s = Snapshot::load();
    let json = snapshot_json();
    let traj = json.get("trajectory");
    let dt = traj.get("dt").num();
    let n_steps = traj.get("n_steps").num() as usize;
    let pe_ref = traj.get("potential_energy").nums();
    let frames = traj.get("frames").arr();
    let (mut state, mut forces, mut pot, cell, e0) = setup(&s);
    assert!((e0 - pe_ref[0]).abs() <= 1e-10 * pe_ref[0].abs());
    let vv = VelocityVerlet::new(dt).unwrap();
    let mut fi = 0;
    let (mut worst_x, mut worst_v) = (0.0f64, 0.0f64);
    assert_eq!(pe_ref.len(), n_steps + 1);
    for (step, &pe) in pe_ref.iter().enumerate().skip(1) {
        let e = vv
            .step(&mut state, &mut forces, &mut pot, Some(&cell))
            .unwrap();
        let de = (e - pe).abs() / pe.abs();
        assert!(de <= 1e-10, "step {step}: PE {e} vs LAMMPS {pe} ({de:e})");
        if fi < frames.len() && frames[fi].get("step").num() as usize == step {
            let x_ref = frames[fi].get("positions").nums();
            let v_ref = frames[fi].get("velocities").nums();
            let v: Vec<f64> = state.p.as_slice().iter().map(|p| p / s.mass).collect();
            let dx = max_abs(state.q.as_slice(), &x_ref);
            let dv = max_abs(&v, &v_ref);
            eprintln!("step {step}: max |dx| = {dx:.2e}, max |dv| = {dv:.2e}");
            assert!(dx <= 1e-8, "step {step}: positions differ by {dx:e}");
            assert!(dv <= 1e-8, "step {step}: velocities differ by {dv:e}");
            (worst_x, worst_v) = (worst_x.max(dx), worst_v.max(dv));
            fi += 1;
        }
    }
    assert_eq!(fi, frames.len(), "not every LAMMPS frame was compared");
    eprintln!("worst over frames: {worst_x:.2e} (x), {worst_v:.2e} (v)");
}

/// Total energy E(t) = K + V sampled after every step for `duration` ħ/Eh at time step `dt`.
fn energy_series(s: &Snapshot, dt: f64, duration: f64) -> Vec<f64> {
    let (mut state, mut forces, mut pot, cell, e0) = setup(s);
    let vv = VelocityVerlet::new(dt).unwrap();
    let steps = (duration / dt).round() as usize;
    let mut out = vec![kinetic(&state) + e0];
    for _ in 0..steps {
        let v = vv
            .step(&mut state, &mut forces, &mut pot, Some(&cell))
            .unwrap();
        out.push(kinetic(&state) + v);
    }
    out
}

/// Standard deviation of the samples about their mean, taking every `stride`-th sample.
fn fluctuation(e: &[f64], stride: usize) -> f64 {
    let pts: Vec<f64> = e.iter().step_by(stride).copied().collect();
    let m = pts.iter().sum::<f64>() / pts.len() as f64;
    (pts.iter().map(|x| (x - m).powi(2)).sum::<f64>() / pts.len() as f64).sqrt()
}

/// I-4: NVE energy conservation scales as dt². From the liquid snapshot (ρ* = 0.8, T* = 1,
/// N = 256, r_c = 2.5σ shifted), the fluctuation of the total energy (standard deviation about its
/// time mean, samples every 0.02 LJ time units, initial sample included) over a fixed duration
/// T = 1 at dt = 0.01 divided by that at dt = 0.005 lies in [3, 5] (dt² gives 4; MILESTONE_3),
/// and the fluctuation at dt = 0.005 is non-zero and below 2e-4 |E|.
///
/// Reference implementation, fluctuation over T = 1: 0.254 (dt = 0.01), 0.0626 (0.005), 0.0159
/// (0.0025), 0.0041 (0.00125): successive ratios 4.06, 3.93, 3.89; over T = 0.5, 2 and 5 the
/// ratios stay in 3.83 … 4.12. The fluctuation at dt = 0.005 is 7.8e-5 |E|. The deviation from
/// E(0) (rather than from the mean) scales less cleanly at small dt (ratio 2.9 between 0.0025
/// and 0.00125 over T = 5): the shifted LJ force still jumps by |F(r_c)| = 0.039 at r_c, and each
/// cutoff crossing adds an O(dt) random-walk error that drifts the mean. That is why the test
/// uses the fluctuation about the mean, at dt where the dt² term dominates.
#[test]
fn energy_error_scales_as_dt_squared() {
    let s = Snapshot::load();
    let duration = 1.0;
    let e1 = energy_series(&s, 0.01, duration);
    let e2 = energy_series(&s, 0.005, duration);
    let (r1, r2) = (fluctuation(&e1, 2), fluctuation(&e2, 4));
    let ratio = r1 / r2;
    eprintln!("energy fluctuation: {r1:.3e} (dt = 0.01), {r2:.3e} (dt = 0.005), ratio {ratio:.3}");
    assert!(
        r2 > 0.0 && r2 < 2e-4 * e2[0].abs(),
        "fluctuation {r2:e}, E0 = {}",
        e2[0]
    );
    assert!(
        (3.0..=5.0).contains(&ratio),
        "ratio {ratio} ({r1:e} / {r2:e})"
    );
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

/// D-3, I-4: 200 steps of NVE (dt = 0.005) from the snapshot in a 1-thread pool and in a 4-thread
/// pool give bitwise identical positions, momenta, forces and energies after every step (the
/// neighbor lists are rebuilt several times along the way).
#[test]
fn two_hundred_steps_bitwise_across_thread_counts() {
    let s = Snapshot::load();
    let run = |threads: usize| {
        pool(threads).install(|| {
            let (mut state, mut forces, mut pot, cell, _) = setup(&s);
            let vv = VelocityVerlet::new(0.005).unwrap();
            let mut energies = Vec::with_capacity(200);
            let mut checks = Vec::new();
            for step in 0..200 {
                energies.push(
                    vv.step(&mut state, &mut forces, &mut pot, Some(&cell))
                        .unwrap(),
                );
                if step % 50 == 49 {
                    checks.push((state.q.clone(), state.p.clone(), forces.clone()));
                }
            }
            (energies, checks, state)
        })
    };
    let (e1, c1, s1) = run(1);
    let (e4, c4, s4) = run(4);
    for (k, (a, b)) in e1.iter().zip(&e4).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "energy after step {}", k + 1);
    }
    let bits = |b: &Beads| b.as_slice().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (k, (a, b)) in c1.iter().zip(&c4).enumerate() {
        assert_eq!(bits(&a.0), bits(&b.0), "positions, checkpoint {k}");
        assert_eq!(bits(&a.1), bits(&b.1), "momenta, checkpoint {k}");
        assert_eq!(bits(&a.2), bits(&b.2), "forces, checkpoint {k}");
    }
    assert_eq!(bits(&s1.q), bits(&s4.q));
    // The run moved the atoms enough that lists were rebuilt (some atom moved > skin/2).
    let moved =
        s1.q.as_slice()
            .chunks(3)
            .zip(s.q.as_slice().chunks(3))
            .map(|(a, b)| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .fold(0.0, f64::max);
    assert!(moved > 0.5 * s.skin, "max displacement {moved}");
}
