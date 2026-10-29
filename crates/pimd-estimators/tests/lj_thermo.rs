//! Thermodynamic comparison with LAMMPS (SPEC P-3, I-4, E-8, D-3): NVE averages of the total
//! potential energy and the pressure of the 256-atom LJ liquid (ρ* = 0.8, T* ≈ 1, r_c = 2.5σ
//! shifted, skin 0.3, dt = 0.005) from the engine (`VelocityVerlet` + `LennardJones` +
//! `pressure_classical`) and from LAMMPS (`data/lj_thermo.json`, generator `data/gen_lj_thermo.py`;
//! LAMMPS version and inputs recorded in the file). Both start from the same microstate (the
//! snapshot in `crates/pimd-potentials/tests/data/lj_snapshot.json`), so they share the total
//! energy and sample the same microcanonical ensemble; lj units = atomic units (ε = σ = m = 1).
//!
//! The long comparison (feature `long-validation`, run in release by `CHECK_LONG=1`) discards
//! 2 000 steps and averages 200 000 (1 000 τ), sampling every 10 steps, exactly as the generator
//! does. Tolerance: |⟨X⟩_engine − ⟨X⟩_LAMMPS| ≤ 5 σ_combined + bias, σ_combined =
//! √(σ_engine² + σ_LAMMPS²), each σ block-averaged (100-sample = 1 000-step blocks, checked
//! against 400-sample blocks, `common::mean_and_sigma`). Bias allowance 0: both codes integrate
//! with the same algorithm (velocity Verlet) and the same dt, so the O(dt²) discretisation bias of
//! the averages is common to both. Validation (reference implementation, release build):
//!   dt = 0.005 (this test): ⟨V⟩ = −1193.396 ± 0.108 vs LAMMPS −1193.294 ± 0.109
//!     (Δ = −0.10 = −0.66 σ_combined); ⟨p⟩ = 1.83291 ± 0.00247 vs LAMMPS 1.83272 ± 0.00228
//!     (Δ = +0.0002 = +0.06 σ_combined).
//!   dt/2 = 0.0025 (same 1 000 τ, samples every 0.05 τ): ⟨V⟩ = −1193.516 ± 0.102,
//!     ⟨p⟩ = 1.83275 ± 0.00242: the change from dt, −0.12 and −0.0002, is unresolved (within
//!     1σ) and below a third of the tolerances 5σ_combined ≈ 0.77 and 0.017.
//! LAMMPS values are from the committed JSON (σ_LAMMPS: 0.109 and 0.0023).
//!
//! The smoke version (default suite) runs the first 400 steps, without equilibration, and compares
//! plain means of the same samples with LAMMPS's: over 400 steps the two trajectories still
//! coincide (reference vs LAMMPS: max |Δx| = 5e-12 at step 400; the separation grows about
//! tenfold per 100 steps and passes 1e-8 near step 800), so the means agree at the trajectory
//! level (measured 8e-15 and 7e-14 relative); tolerance 1e-9 relative.

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

mod common;

use lammps::{Json, Snapshot};
use pimd_estimators::pressure_classical;
use pimd_integrators::{RingState, VelocityVerlet};
use pimd_potentials::{LennardJones, LjParams, Potential};

const THERMO_JSON: &str = include_str!("data/lj_thermo.json");

/// Runs NVE from the snapshot for `n_equil + n_prod` steps of `dt`; after every `sample`-th
/// production step records the total potential energy (returned by `step`) and the pressure
/// (E-8, from the P = 1 virial of one extra `evaluate` at the current positions: `step` returns
/// only the energy).
fn run(dt: f64, n_equil: usize, n_prod: usize, sample: usize) -> (Vec<f64>, Vec<f64>) {
    let s = Snapshot::load();
    let cell = s.cell();
    let mut pot = LennardJones::new(LjParams {
        epsilon: s.epsilon,
        sigma: s.sigma,
        cutoff: s.cutoff,
        skin: s.skin,
        shift: true,
    })
    .unwrap();
    let mut forces = pot.evaluate(&s.q, Some(&cell), None).unwrap().forces;
    let mut state = RingState {
        q: s.q.clone(),
        p: s.p.clone(),
        masses: s.masses(),
    };
    let vv = VelocityVerlet::new(dt).unwrap();
    let (mut pe, mut press) = (Vec::new(), Vec::new());
    for step in 1..=n_equil + n_prod {
        let e = vv
            .step(&mut state, &mut forces, &mut pot, Some(&cell))
            .unwrap();
        if step > n_equil && (step - n_equil).is_multiple_of(sample) {
            let w = pot
                .evaluate(&state.q, Some(&cell), None)
                .unwrap()
                .virial
                .unwrap();
            pe.push(e);
            press.push(pressure_classical(&state.masses, &state.p, w, &cell).unwrap());
        }
    }
    (pe, press)
}

/// P-3, I-4, E-8 (smoke): the first 400 steps from the snapshot, samples every 10 steps (40
/// samples): mean total potential energy and mean pressure equal LAMMPS's over the same samples to
/// 1e-9 relative (trajectory-level agreement, see the module comment).
#[test]
fn smoke_first_400_steps_match_lammps() {
    let j = Json::parse(THERMO_JSON);
    let smoke = j.get("smoke");
    let n = smoke.get("n_steps").num() as usize;
    let dt = j.at(&["protocol", "dt"]).num();
    let sample = j.at(&["protocol", "sample_every"]).num() as usize;
    let (pe, press) = run(dt, 0, n, sample);
    assert_eq!(pe.len(), smoke.get("n_samples").num() as usize);
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    for (what, x, want) in [
        (
            "potential energy",
            &pe,
            smoke.get("potential_energy_mean").num(),
        ),
        ("pressure", &press, smoke.get("pressure_mean").num()),
    ] {
        let m = mean(x);
        let rel = (m - want).abs() / want.abs();
        eprintln!("smoke {what}: engine {m} LAMMPS {want} rel {rel:.2e}");
        assert!(rel <= 1e-9, "{what}: {m} vs LAMMPS {want} ({rel:e})");
    }
}

/// P-3, I-4, E-8 (long validation): equilibrated NVE averages of the total potential energy and
/// the pressure agree with LAMMPS within 5σ_combined (module comment).
#[cfg(feature = "long-validation")]
#[test]
fn nve_averages_match_lammps() {
    let j = Json::parse(THERMO_JSON);
    let pr = j.get("protocol");
    let dt = pr.get("dt").num();
    let n_equil = pr.get("n_equil").num() as usize;
    let n_prod = pr.get("n_prod").num() as usize;
    let sample = pr.get("sample_every").num() as usize;
    let (pe, press) = run(dt, n_equil, n_prod, sample);
    for (what, x, key) in [
        ("potential energy", &pe, "potential_energy"),
        ("pressure", &press, "pressure"),
    ] {
        let reference = j.get(key);
        let block = reference.get("block_samples").num() as usize;
        assert_eq!(x.len(), reference.get("n_samples").num() as usize);
        let (m, sigma) = common::mean_and_sigma(x, block, what);
        let (m_ref, s_ref) = (reference.get("mean").num(), reference.get("sigma").num());
        let combined = (sigma * sigma + s_ref * s_ref).sqrt();
        eprintln!(
            "{what}: engine {m:.6} ± {sigma:.2e}, LAMMPS {m_ref:.6} ± {s_ref:.2e}, \
             diff {:+.3e} ({:.2} σ_combined)",
            m - m_ref,
            (m - m_ref) / combined
        );
        assert!(
            (m - m_ref).abs() <= 5.0 * combined,
            "{what}: engine {m} ± {sigma}, LAMMPS {m_ref} ± {s_ref}"
        );
    }
}
