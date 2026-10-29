//! Energy estimators E-1 … E-5 and the ring-polymer Hamiltonian H_P (RP-3).
//!
//! All results are totals over atoms and Cartesian components, in Eh. Positions `q` and forces
//! have shape `[P, N, 3]` (bohr, Eh/bohr), `masses` length N (m_e), `energies` length P (Eh),
//! β in Eh⁻¹. Every function panics if the input shapes disagree (Q4).

use pimd_core::Beads;
use pimd_integrators::RingState;

/// E_spring = Σ_j Σ_i ½ m_i ω_P² |q_{j,i} − q_{j+1,i}|², with ω_P = P/β and cyclic j (E-5).
///
/// Panics if `masses.len()` differs from N.
pub fn spring_energy(q: &Beads, masses: &[f64], beta: f64) -> f64 {
    assert_masses(q, masses);
    let p = q.n_beads();
    let wp = p as f64 / beta;
    let mut sum = 0.0;
    for j in 0..p {
        let next = (j + 1) % p;
        for (i, &m) in masses.iter().enumerate() {
            let (a, b) = (q.get(j, i), q.get(next, i));
            let d2: f64 = (0..3).map(|d| (a[d] - b[d]).powi(2)).sum();
            sum += 0.5 * m * wp * wp * d2;
        }
    }
    sum
}

/// Primitive kinetic estimator K_prim = 3NP/(2β) − E_spring/P (E-1).
///
/// Panics if `masses.len()` differs from N.
pub fn kinetic_primitive(q: &Beads, masses: &[f64], beta: f64) -> f64 {
    let (p, n) = (q.n_beads() as f64, q.n_atoms() as f64);
    3.0 * n * p / (2.0 * beta) - spring_energy(q, masses, beta) / p
}

/// Virial kinetic estimator K_vir = 3N/(2β) − (1/(2P)) Σ_j Σ_i (q_{j,i} − q̄_i)·F_{j,i} (E-2),
/// with `forces` = −∇V (S-3, E-6).
///
/// Panics if `forces` and `q` differ in shape.
pub fn kinetic_virial(q: &Beads, forces: &Beads, beta: f64) -> f64 {
    assert_same_shape(q, forces, "forces");
    let (p, n) = (q.n_beads(), q.n_atoms());
    let centroid = q.centroid();
    let mut sum = 0.0;
    for j in 0..p {
        for (i, c) in centroid.iter().enumerate() {
            let (x, f) = (q.get(j, i), forces.get(j, i));
            sum += (0..3).map(|d| (x[d] - c[d]) * f[d]).sum::<f64>();
        }
    }
    3.0 * n as f64 / (2.0 * beta) - sum / (2.0 * p as f64)
}

/// Bead-averaged potential V̄ = (1/P) Σ_j V(q_j) (E-3), with P = `energies.len()`.
pub fn potential_mean(energies: &[f64]) -> f64 {
    energies.iter().sum::<f64>() / energies.len() as f64
}

/// E_prim = K_prim + V̄ (E-4).
///
/// Panics if `masses.len()` differs from N or `energies.len()` from P.
pub fn total_primitive(q: &Beads, masses: &[f64], beta: f64, energies: &[f64]) -> f64 {
    assert_energies(q, energies);
    kinetic_primitive(q, masses, beta) + potential_mean(energies)
}

/// E_vir = K_vir + V̄ (E-4).
///
/// Panics if `forces` and `q` differ in shape or `energies.len()` differs from P.
pub fn total_virial(q: &Beads, forces: &Beads, beta: f64, energies: &[f64]) -> f64 {
    assert_energies(q, energies);
    kinetic_virial(q, forces, beta) + potential_mean(energies)
}

/// Ring-polymer Hamiltonian H_P = Σ_j Σ_i p²/(2 m_i) + E_spring + Σ_j V(q_j) (RP-3), with
/// `energies[j]` = V(q_j).
///
/// Panics if `state.p` and `state.q` differ in shape, `state.masses.len()` differs from N or
/// `energies.len()` from P.
pub fn hamiltonian(state: &RingState, beta: f64, energies: &[f64]) -> f64 {
    let (q, p) = (&state.q, &state.p);
    assert_same_shape(q, p, "p");
    assert_energies(q, energies);
    let spring = spring_energy(q, &state.masses, beta);
    let mut kinetic = 0.0;
    for j in 0..p.n_beads() {
        for (i, &m) in state.masses.iter().enumerate() {
            kinetic += p.get(j, i).iter().map(|x| x * x).sum::<f64>() / (2.0 * m);
        }
    }
    kinetic + spring + energies.iter().sum::<f64>()
}

pub(crate) fn assert_masses(q: &Beads, masses: &[f64]) {
    assert_eq!(
        masses.len(),
        q.n_atoms(),
        "{} masses for {} atoms",
        masses.len(),
        q.n_atoms()
    );
}

fn assert_energies(q: &Beads, energies: &[f64]) {
    assert_eq!(
        energies.len(),
        q.n_beads(),
        "{} energies for {} beads",
        energies.len(),
        q.n_beads()
    );
}

fn assert_same_shape(q: &Beads, other: &Beads, name: &str) {
    assert!(
        q.n_beads() == other.n_beads() && q.n_atoms() == other.n_atoms(),
        "q has shape [{}, {}, 3], {name} [{}, {}, 3]",
        q.n_beads(),
        q.n_atoms(),
        other.n_beads(),
        other.n_atoms()
    );
}
