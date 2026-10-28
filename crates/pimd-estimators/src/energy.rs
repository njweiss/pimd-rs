//! Energy estimators E-1 … E-5 and the ring-polymer Hamiltonian H_P (RP-3).
//!
//! All results are totals over atoms and Cartesian components, in Eh. Positions `q` and forces
//! have shape `[P, N, 3]` (bohr, Eh/bohr), `masses` length N (m_e), `energies` length P (Eh),
//! β in Eh⁻¹. Every function panics if the input shapes disagree (Q4).

use pimd_core::Beads;
use pimd_integrators::RingState;

/// E_spring = Σ_j Σ_i ½ m_i ω_P² |q_{j,i} − q_{j+1,i}|², with ω_P = P/β and cyclic j (E-5).
pub fn spring_energy(q: &Beads, masses: &[f64], beta: f64) -> f64 {
    let _ = (q, masses, beta);
    unimplemented!()
}

/// Primitive kinetic estimator K_prim = 3NP/(2β) − E_spring/P (E-1).
pub fn kinetic_primitive(q: &Beads, masses: &[f64], beta: f64) -> f64 {
    let _ = (q, masses, beta);
    unimplemented!()
}

/// Virial kinetic estimator K_vir = 3N/(2β) − (1/(2P)) Σ_j Σ_i (q_{j,i} − q̄_i)·F_{j,i} (E-2),
/// with `forces` = −∇V (S-3, E-6).
pub fn kinetic_virial(q: &Beads, forces: &Beads, beta: f64) -> f64 {
    let _ = (q, forces, beta);
    unimplemented!()
}

/// Bead-averaged potential V̄ = (1/P) Σ_j V(q_j) (E-3).
pub fn potential_mean(energies: &[f64]) -> f64 {
    let _ = energies;
    unimplemented!()
}

/// E_prim = K_prim + V̄ (E-4).
pub fn total_primitive(q: &Beads, masses: &[f64], beta: f64, energies: &[f64]) -> f64 {
    let _ = (q, masses, beta, energies);
    unimplemented!()
}

/// E_vir = K_vir + V̄ (E-4).
pub fn total_virial(q: &Beads, forces: &Beads, beta: f64, energies: &[f64]) -> f64 {
    let _ = (q, forces, beta, energies);
    unimplemented!()
}

/// Ring-polymer Hamiltonian H_P = Σ_j Σ_i p²/(2 m_i) + E_spring + Σ_j V(q_j) (RP-3), with
/// `energies[j]` = V(q_j).
pub fn hamiltonian(state: &RingState, beta: f64, energies: &[f64]) -> f64 {
    let _ = (state, beta, energies);
    unimplemented!()
}
