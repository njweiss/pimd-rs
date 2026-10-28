//! Tests for the energy estimators and H_P (SPEC E-1 … E-6, RP-3, RP-6, S-3), against
//! hand-computed examples and independent formulas written in the test.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

use std::panic::{AssertUnwindSafe, catch_unwind};

use pimd_core::Beads;
use pimd_estimators::{
    hamiltonian, kinetic_primitive, kinetic_virial, potential_mean, spring_energy, total_primitive,
    total_virial,
};
use pimd_integrators::RingState;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

/// The hand example: P = 2, N = 1, m = 2, β = 2 (so ω_P = P/β = 1, RP-2).
/// q_0 = (1, 2, 0), q_1 = (0, 1, 1); F_0 = (0.5, −1, 2), F_1 = (1, 0, −0.5);
/// p_0 = (1, 0, 0), p_1 = (0, 2, 0); V = (0.25, 0.75). All values are exact in binary.
fn hand() -> (Beads, Beads, Vec<f64>, f64, Vec<f64>, Beads) {
    let q = Beads::from_vec(2, 1, vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
    let f = Beads::from_vec(2, 1, vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5]).unwrap();
    let p = Beads::from_vec(2, 1, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).unwrap();
    (q, f, vec![2.0], 2.0, vec![0.25, 0.75], p)
}

/// E-5, RP-2, RP-1: hand example. q_0 − q_1 = (1, 1, −1), |·|² = 3. With cyclic indexing and
/// P = 2 both links j = 0 (0→1) and j = 1 (1→0) contribute ½·m·ω_P²·3 = ½·2·1·3 = 3, so
/// E_spring = 6.
#[test]
fn spring_energy_hand_example() {
    let (q, _, m, beta, _, _) = hand();
    let e = spring_energy(&q, &m, beta);
    assert!(close(e, 6.0, 1e-14), "{e}");
}

/// E-1: K_prim = 3NP/(2β) − E_spring/P = 3·1·2/(2·2) − 6/2 = 1.5 − 3 = −1.5.
#[test]
fn kinetic_primitive_hand_example() {
    let (q, _, m, beta, _, _) = hand();
    let k = kinetic_primitive(&q, &m, beta);
    assert!(close(k, -1.5, 1e-14), "{k}");
}

/// E-2, E-6: q̄ = (0.5, 1.5, 0.5); q_0 − q̄ = (0.5, 0.5, −0.5), q_1 − q̄ = (−0.5, −0.5, 0.5).
/// (q_0 − q̄)·F_0 = 0.25 − 0.5 − 1 = −1.25; (q_1 − q̄)·F_1 = −0.5 + 0 − 0.25 = −0.75; sum −2.
/// K_vir = 3N/(2β) − (1/(2P))·(−2) = 0.75 + 0.5 = 1.25.
#[test]
fn kinetic_virial_hand_example() {
    let (q, f, _, beta, _, _) = hand();
    let k = kinetic_virial(&q, &f, beta);
    assert!(close(k, 1.25, 1e-14), "{k}");
}

/// E-3: V̄ = (0.25 + 0.75)/2 = 0.5; a second case (1, 2, 6)/3 = 3; P = 1 gives the single value.
#[test]
fn potential_mean_hand_examples() {
    assert!(close(potential_mean(&[0.25, 0.75]), 0.5, 1e-15));
    assert!(close(potential_mean(&[1.0, 2.0, 6.0]), 3.0, 1e-15));
    assert!(close(potential_mean(&[-4.5]), -4.5, 1e-15));
}

/// E-4: E_prim = K_prim + V̄ = −1.5 + 0.5 = −1.0; E_vir = K_vir + V̄ = 1.25 + 0.5 = 1.75.
#[test]
fn totals_hand_example() {
    let (q, f, m, beta, v, _) = hand();
    let ep = total_primitive(&q, &m, beta, &v);
    let ev = total_virial(&q, &f, beta, &v);
    assert!(close(ep, -1.0, 1e-14), "{ep}");
    assert!(close(ev, 1.75, 1e-14), "{ev}");
}

/// RP-3: H_P = Σ p²/(2m) + E_spring + Σ_j V(q_j) = (1 + 4)/(2·2) + 6 + (0.25 + 0.75)
/// = 1.25 + 6 + 1 = 8.25 (the potential enters as the bead sum, not the mean).
#[test]
fn hamiltonian_hand_example() {
    let (q, _, m, beta, v, p) = hand();
    let s = RingState { q, p, masses: m };
    let h = hamiltonian(&s, beta, &v);
    assert!(close(h, 8.25, 1e-14), "{h}");
}

/// SplitMix64, test-only seeded generator.
struct SplitMix64(u64);

impl SplitMix64 {
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        lo + (hi - lo) * ((z >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }

    fn beads(&mut self, p: usize, n: usize) -> Beads {
        Beads::from_vec(
            p,
            n,
            (0..p * n * 3).map(|_| self.range(-2.0, 2.0)).collect(),
        )
        .unwrap()
    }
}

/// E-1, E-2, E-5, RP-3: P = 3 and 5, N = 3 with distinct masses, random q, F, p, V, β = 0.8,
/// against the SPEC formulas written as plain loops here (cyclic springs with per-atom mass,
/// centroid-referenced virial), to 1e-12 relative.
#[test]
fn random_cases_match_spec_formulas() {
    let masses = [1.0, 4.0, 0.5];
    let beta = 0.8;
    let mut rng = SplitMix64(7);
    for p in [3usize, 5] {
        let n = masses.len();
        let (q, f, pm) = (rng.beads(p, n), rng.beads(p, n), rng.beads(p, n));
        let v: Vec<f64> = (0..p).map(|_| rng.range(-1.0, 1.0)).collect();
        let (pf, nf) = (p as f64, n as f64);
        let wp = pf / beta;
        let mut es = 0.0;
        let mut kin = 0.0;
        let mut vir = 0.0;
        for i in 0..n {
            let mut cen = [0.0; 3];
            for j in 0..p {
                for d in 0..3 {
                    cen[d] += q.get(j, i)[d] / pf;
                }
            }
            for j in 0..p {
                let (a, b) = (q.get(j, i), q.get((j + 1) % p, i));
                for d in 0..3 {
                    es += 0.5 * masses[i] * wp * wp * (a[d] - b[d]).powi(2);
                    kin += pm.get(j, i)[d].powi(2) / (2.0 * masses[i]);
                    vir += (a[d] - cen[d]) * f.get(j, i)[d];
                }
            }
        }
        let vbar = v.iter().sum::<f64>() / pf;
        let kp = 3.0 * nf * pf / (2.0 * beta) - es / pf;
        let kv = 3.0 * nf / (2.0 * beta) - vir / (2.0 * pf);
        let h = kin + es + v.iter().sum::<f64>();
        assert!(
            close(spring_energy(&q, &masses, beta), es, 1e-12),
            "P = {p}: spring"
        );
        assert!(
            close(kinetic_primitive(&q, &masses, beta), kp, 1e-12),
            "P = {p}: K_prim"
        );
        assert!(
            close(kinetic_virial(&q, &f, beta), kv, 1e-12),
            "P = {p}: K_vir"
        );
        assert!(
            close(total_primitive(&q, &masses, beta, &v), kp + vbar, 1e-12),
            "P = {p}"
        );
        assert!(
            close(total_virial(&q, &f, beta, &v), kv + vbar, 1e-12),
            "P = {p}"
        );
        let s = RingState {
            q,
            p: pm,
            masses: masses.to_vec(),
        };
        assert!(close(hamiltonian(&s, beta, &v), h, 1e-12), "P = {p}: H_P");
    }
}

/// E-2: the virial estimator is invariant under a rigid translation of every bead of an atom
/// (q − q̄ is unchanged), here by (10, −20, 5), with the same forces.
#[test]
fn virial_is_translation_invariant() {
    let mut rng = SplitMix64(9);
    let (q, f) = (rng.beads(4, 2), rng.beads(4, 2));
    let mut shifted = q.clone();
    for j in 0..4 {
        for i in 0..2 {
            let x = q.get(j, i);
            shifted.set(j, i, [x[0] + 10.0, x[1] - 20.0, x[2] + 5.0]);
        }
    }
    let (a, b) = (
        kinetic_virial(&q, &f, 1.3),
        kinetic_virial(&shifted, &f, 1.3),
    );
    assert!(close(b, a, 1e-12), "{a} vs {b}");
}

/// RP-6, E-1, E-2, E-5: for P = 1 there are no springs and q = q̄, so E_spring = 0 and both
/// kinetic estimators equal 3N/(2β) at any positions and forces (N = 3, β = 0.6: 7.5).
#[test]
fn p1_kinetic_estimators_are_classical() {
    let mut rng = SplitMix64(11);
    for _ in 0..5 {
        let (q, f) = (rng.beads(1, 3), rng.beads(1, 3));
        let m = [1.0, 2.0, 3.0];
        assert!(spring_energy(&q, &m, 0.6).abs() <= 1e-15);
        assert!(close(kinetic_primitive(&q, &m, 0.6), 7.5, 1e-14));
        assert!(close(kinetic_virial(&q, &f, 0.6), 7.5, 1e-14));
    }
}

/// Runs `f` and returns true if it panicked with something other than a stub's
/// `unimplemented!` ("not implemented"), so these tests fail until the estimators exist.
fn panics_for_real(f: impl FnOnce()) -> bool {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => false,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            !msg.contains("not implemented")
        }
    }
}

/// Q4 decision (estimators panic on shape mismatch): masses.len() != N, forces shape differing
/// from q (bead or atom count), and energies.len() != P each panic.
#[test]
fn shape_mismatches_panic() {
    let q = Beads::zeros(2, 2);
    let f_beads = Beads::zeros(3, 2);
    let f_atoms = Beads::zeros(2, 1);
    let f = Beads::zeros(2, 2);
    let m = [1.0, 1.0];
    assert!(panics_for_real(|| {
        spring_energy(&q, &[1.0], 1.0);
    }));
    assert!(panics_for_real(|| {
        kinetic_primitive(&q, &[1.0, 1.0, 1.0], 1.0);
    }));
    assert!(panics_for_real(|| {
        kinetic_virial(&q, &f_beads, 1.0);
    }));
    assert!(panics_for_real(|| {
        kinetic_virial(&q, &f_atoms, 1.0);
    }));
    assert!(panics_for_real(|| {
        total_primitive(&q, &m, 1.0, &[0.0; 3]);
    }));
    assert!(panics_for_real(|| {
        total_virial(&q, &f, 1.0, &[0.0]);
    }));
    let s = RingState {
        q: q.clone(),
        p: Beads::zeros(2, 2),
        masses: m.to_vec(),
    };
    assert!(panics_for_real(|| {
        hamiltonian(&s, 1.0, &[0.0]);
    }));
}
