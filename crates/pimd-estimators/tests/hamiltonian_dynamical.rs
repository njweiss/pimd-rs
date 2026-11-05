//! Tests for `pimd_estimators::hamiltonian_dynamical` (MILESTONE_7 M7b required test 4; SPEC Y-6,
//! RP-3, RP-4, E-5, T-3, I-1, I-2).
//!
//! Expectations: a hand evaluation of H̃_P (worked out below); `hamiltonian` (merged M2 code) for
//! s ≡ 1, since the RP-4 transform is orthogonal; the O(dt²) scaling of the energy error of
//! BAOAB without the O step (T-3), which holds for the Hamiltonian that the A step propagates
//! exactly, H̃_P with the Y-6 masses.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

#[path = "../../pimd-integrators/tests/common/mod.rs"]
mod common;
#[path = "../../pimd-integrators/tests/m7_common/mod.rs"]
mod m7_common;

use common::{SplitMix64, random_beads};
use m7_common::{CLUSTER_BETA, cluster_state, lj};
use pimd_core::{Beads, NormalModes, Rng, Transform};
use pimd_estimators::{hamiltonian, hamiltonian_dynamical};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};

fn beads(p: usize, n: usize, rows: &[[[f64; 3]; 2]]) -> Beads {
    let mut b = Beads::zeros(p, n);
    for (j, r) in rows.iter().enumerate() {
        for i in 0..n {
            b.set(j, i, r[i]);
        }
    }
    b
}

/// The hand case: P = 4, N = 2, β = 2 (ω_P = 2), masses (2, 0.5), mass factors
/// s = (1, 0.5, 0.25, 0.125), energies (0.5, −1, 2, 0.25).
fn hand_case() -> (RingState, f64, Vec<f64>, Vec<f64>) {
    // Bead momenta p_j (j = 0 … 3) of atom 0 and atom 1.
    let p = beads(
        4,
        2,
        &[
            [[3.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
            [[1.0, 1.0, 2.0], [-1.0, 0.0, 0.0]],
            [[-1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 1.0, -2.0], [-1.0, 0.0, 0.0]],
        ],
    );
    // Positions: atom 0 x = (0, 1, 0, 1), atom 1 z = (0, 0, 0, 2), rest 0.
    let q = beads(
        4,
        2,
        &[
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        ],
    );
    let state = RingState {
        q,
        p,
        masses: vec![2.0, 0.5],
    };
    (
        state,
        2.0,
        vec![0.5, -1.0, 2.0, 0.25],
        vec![1.0, 0.5, 0.25, 0.125],
    )
}

/// Y-6 (hand value). RP-4 for P = 4: columns C_{j0} = 1/2, C_{j1} = (1/√2)(1, 0, −1, 0),
/// C_{j2} = (1/√2)(0, 1, 0, −1), C_{j3} = (1/2)(1, −1, 1, −1). Normal-mode momenta
/// p̃_k = Σ_j C_{jk} p_j:
///   atom 0 x: p = (3, 1, −1, 1) → p̃ = (2, 2√2, 0, 0);  y: (1, 1, 1, 1) → (2, 0, 0, 0);
///            z: (0, 2, 0, −2) → (0, 0, 2√2, 0);
///   atom 1 x: (1, −1, 1, −1) → (0, 0, 0, 2);          y: (2, 0, 0, 0) → (1, √2, 0, 1); z: 0.
/// Kinetic Σ_k Σ_i |p̃_{k,i}|²/(2 m_i s_k):
///   atom 0 (m = 2): k0 (4 + 4)/4 = 2; k1 8/2 = 4; k2 8/1 = 8 → 14;
///   atom 1 (m = 0.5): k0 1/1 = 1; k1 2/0.5 = 4; k3 (4 + 1)/0.125 = 40 → 45; total 59.
/// Springs (E-5) ½ m ω_P² Σ_j |q_j − q_{j+1}|², ω_P² = 4: atom 0: ½·2·4·4 = 16; atom 1: ½·0.5·4·8 = 8;
/// total 24. Σ V_j = 1.75. H̃_P = 59 + 24 + 1.75 = 84.75 (1e-14 relative, both transforms).
/// With s ≡ 1 the kinetic part is the bead sum Σ p²/(2m) = 24/4 + 8/1 = 14, so H̃_P = 39.75, which
/// is also `hamiltonian` (merged M2 code) of the same state.
#[test]
fn hand_value() {
    let (state, beta, energies, s) = hand_case();
    for transform in [Transform::Matrix, Transform::Fft] {
        let modes = NormalModes::with_transform(4, beta, transform).unwrap();
        let h = hamiltonian_dynamical(&state, beta, &energies, &modes, &s);
        assert!((h - 84.75).abs() <= 1e-14 * 84.75, "{transform:?}: {h}");
        let h1 = hamiltonian_dynamical(&state, beta, &energies, &modes, &[1.0; 4]);
        assert!(
            (h1 - 39.75).abs() <= 1e-14 * 39.75,
            "{transform:?}, s = 1: {h1}"
        );
    }
    let h = hamiltonian(&state, beta, &energies);
    assert!((h - 39.75).abs() <= 1e-14 * 39.75, "hamiltonian: {h}");
}

/// Y-6 ("without Ω (s ≡ 1) nothing changes"): for s ≡ 1 `hamiltonian_dynamical` equals
/// `hamiltonian` to 1e-12 relative on random states, P = 1, 2, 3, 4, 7, 8, N = 3, both
/// transforms.
#[test]
fn unit_factors_equal_hamiltonian() {
    let mut sm = SplitMix64(301);
    let masses = vec![1.0, 4.0, 1836.0];
    for p in [1, 2, 3, 4, 7, 8] {
        for transform in [Transform::Matrix, Transform::Fft] {
            let beta = 1.7;
            let state = RingState {
                q: random_beads(&mut sm, p, 3, -1.0, 1.0),
                p: random_beads(&mut sm, p, 3, -3.0, 3.0),
                masses: masses.clone(),
            };
            let energies: Vec<f64> = (0..p).map(|_| sm.range(-1.0, 1.0)).collect();
            let modes = NormalModes::with_transform(p, beta, transform).unwrap();
            let a = hamiltonian_dynamical(&state, beta, &energies, &modes, &vec![1.0; p]);
            let b = hamiltonian(&state, beta, &energies);
            assert!(
                (a - b).abs() <= 1e-12 * b.abs(),
                "P = {p}, {transform:?}: {a} vs {b}"
            );
        }
    }
}

/// Y-6 (shape rule of the brief): `mass_factors.len()` ≠ P panics.
#[test]
#[should_panic]
fn wrong_factor_count_panics() {
    let (state, beta, energies, _) = hand_case();
    let modes = NormalModes::new(4, beta).unwrap();
    hamiltonian_dynamical(&state, beta, &energies, &modes, &[1.0, 0.5, 0.25]);
}

/// Y-6 (shape rules of `hamiltonian`, Q4): an energy count ≠ P panics.
#[test]
#[should_panic]
fn wrong_energy_count_panics() {
    let (state, beta, energies, s) = hand_case();
    let modes = NormalModes::new(4, beta).unwrap();
    hamiltonian_dynamical(&state, beta, &energies[..3], &modes, &s);
}

/// Y-6 (shape rules of `hamiltonian`): momenta of a different shape than the positions panic.
#[test]
#[should_panic]
fn wrong_momentum_shape_panics() {
    let (mut state, beta, energies, s) = hand_case();
    state.p = Beads::zeros(4, 1);
    let modes = NormalModes::new(4, beta).unwrap();
    hamiltonian_dynamical(&state, beta, &energies, &modes, &s);
}

/// Mean over the run of |H̃_P(t) − H̃_P(0)| sampled every 0.05 ħ/Eh, for the 8-atom LJ cluster
/// (P = 4, β = 4: ω_P = 1, ω_k = √2, √2, 2), `Off`, Ω = 3 (s = 2/9, 2/9, 4/9), momenta first
/// redrawn with Ω (amended Y-2, so the internal modes start at the temperature of the dynamical
/// masses), total time 5 ħ/Eh at time step `dt`.
fn mean_energy_error(dt: f64) -> f64 {
    let omega = 3.0;
    let ig = RingPolymerIntegrator::new(dt, CLUSTER_BETA, 4, Thermostat::Off)
        .unwrap()
        .with_adiabatic_frequency(omega)
        .unwrap();
    let mut sim = RingPolymerSim::new(cluster_state(4, 302), None, ig, Box::new(lj())).unwrap();
    let h = |sim: &RingPolymerSim| {
        let ig = sim.integrator();
        hamiltonian_dynamical(
            sim.state(),
            CLUSTER_BETA,
            sim.energies(),
            ig.modes(),
            ig.mass_factors(),
        )
    };
    let mut rng = Rng::from_seed(1);
    sim.resample_momenta(&mut rng);
    let h0 = h(&sim);
    let every = (0.05 / dt).round() as usize;
    let n = (5.0 / dt).round() as usize;
    let (mut sum, mut count) = (0.0, 0);
    for s in 1..=n {
        sim.step(&mut rng).unwrap();
        if s % every == 0 {
            sum += (h(&sim) - h0).abs();
            count += 1;
        }
    }
    sum / count as f64
}

/// Y-6, T-3, I-1, I-2: under `Off` with Ω (LJ cluster, P = 4), H̃_P is the energy of the BAOAB
/// dynamics (the A step is the exact flow of the dynamical-mass free ring polymer, the B step the
/// exact flow of V), so its error is the O(dt²) shadow-Hamiltonian error: the mean |ΔH̃_P| over the
/// same trajectory time falls by a factor in [3.5, 4.5] when dt halves (0.01 → 0.005; 4·(1 +
/// O(dt²))). With physical masses in the kinetic term, or the bead kinetic energy, ΔH would not
/// shrink with dt.
#[test]
fn energy_error_scales_as_dt_squared_with_omega() {
    let (e1, e2) = (mean_energy_error(0.01), mean_energy_error(0.005));
    let ratio = e1 / e2;
    eprintln!("mean |ΔH̃|: dt {e1:.4e}, dt/2 {e2:.4e}, ratio {ratio:.3}");
    assert!((3.5..=4.5).contains(&ratio), "ratio {ratio}");
}
