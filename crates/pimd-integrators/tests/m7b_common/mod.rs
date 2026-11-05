//! Test-only helpers for the M7b (PA-CMD) integrator tests (MILESTONE_7 "M7b brief"; SPEC Y-6,
//! I-2, T-1, T-2, T-4, Y-2).
//!
//! Everything is written from the SPEC formulas on top of the locked M2 helpers in `common` (RP-4
//! matrix and frequencies): the Y-6 mass factors s_k, the I-2 free propagation with mass m_i·s_k
//! and dynamical frequency Ω_k, the T-1 O step with noise variance (1 − c_k²)·m_i·s_k·P/β. Nothing
//! here calls the code under test to compute an expectation.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

use pimd_core::{Beads, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, Thermostat};

use crate::common::{ref_freqs, ref_matrix, to_beads, to_modes};
use crate::m7_common::{CLUSTER_BETA, CLUSTER_DT, cluster_state, lj};

/// Y-6: s_0 = 1 and s_k = (ω_k/Ω)² for ω_k > 0 (ω_k from RP-2/RP-4), aligned with the RP-4
/// columns.
pub fn ref_mass_factors(p: usize, beta: f64, omega: f64) -> Vec<f64> {
    ref_freqs(p, beta)
        .iter()
        .map(|&w| if w > 0.0 { (w / omega).powi(2) } else { 1.0 })
        .collect()
}

/// Y-6: the dynamical frequencies Ω_k = ω_k/√s_k (Ω for k ≥ 1, 0 for the centroid).
pub fn ref_dyn_freqs(p: usize, omega: f64) -> Vec<f64> {
    (0..p).map(|k| if k == 0 { 0.0 } else { omega }).collect()
}

/// I-2 with Y-6: per mode k and component, mass m_i·s_k and frequency Ω_k:
/// Ω_k > 0: q̃' = cos(Ω_k t) q̃ + sin(Ω_k t)/(m_i s_k Ω_k) p̃, p̃' = −m_i s_k Ω_k sin(Ω_k t) q̃ + cos(Ω_k t) p̃;
/// Ω_k = 0 (centroid): q̃' = q̃ + t p̃/m_i.
pub fn ref_free_propagate_dyn(
    q: &Beads,
    p: &Beads,
    masses: &[f64],
    beta: f64,
    omega: f64,
    t: f64,
) -> (Beads, Beads) {
    let nb = q.n_beads();
    let c = ref_matrix(nb);
    let s = ref_mass_factors(nb, beta, omega);
    let w = ref_dyn_freqs(nb, omega);
    let mut qt = to_modes(&c, q);
    let mut pt = to_modes(&c, p);
    for k in 0..nb {
        for (i, &m0) in masses.iter().enumerate() {
            let m = m0 * s[k];
            let (mut a, mut b) = (qt.get(k, i), pt.get(k, i));
            for d in 0..3 {
                let (q0, p0) = (a[d], b[d]);
                if w[k] > 0.0 {
                    let (sn, co) = (w[k] * t).sin_cos();
                    a[d] = co * q0 + sn / (m * w[k]) * p0;
                    b[d] = -m * w[k] * sn * q0 + co * p0;
                } else {
                    a[d] = q0 + t * p0 / m;
                }
            }
            qt.set(k, i, a);
            pt.set(k, i, b);
        }
    }
    (to_beads(&c, &qt), to_beads(&c, &pt))
}

/// c_k = exp(−γ_k dt) with the Y-6 friction: γ_0 = `gamma0`, γ_k = `scale`·Ω for k ≥ 1
/// (PILE-L: scale 2, T-2; TRPMD: scale 2λ, T-4).
pub fn ref_c_dyn(p: usize, omega: f64, dt: f64, gamma0: f64, scale: f64) -> Vec<f64> {
    ref_dyn_freqs(p, omega)
        .iter()
        .enumerate()
        .map(|(k, &w)| {
            let g = if k == 0 { gamma0 } else { scale * w };
            (-g * dt).exp()
        })
        .collect()
}

/// The T-1 O step with Y-6: p̃_k ← c_k p̃_k + sqrt((1 − c_k²)·m_i·s_k·P/β)·ξ, ξ drawn k outermost,
/// then atom i, then d (one normal per (k, i, d)).
pub fn ref_o_step_dyn(
    p: &Beads,
    masses: &[f64],
    beta: f64,
    c: &[f64],
    s: &[f64],
    rng: &mut Rng,
) -> Beads {
    let nb = p.n_beads();
    let pf = nb as f64;
    let mat = ref_matrix(nb);
    let mut pt = to_modes(&mat, p);
    for k in 0..nb {
        for (i, &m) in masses.iter().enumerate() {
            let mut v = pt.get(k, i);
            for x in &mut v {
                let xi = rng.normal();
                *x = c[k] * *x + ((1.0 - c[k] * c[k]) * m * s[k] * pf / beta).sqrt() * xi;
            }
            pt.set(k, i, v);
        }
    }
    to_beads(&mat, &pt)
}

/// The amended Y-2 redraw written out: z_{j,i,d} = sqrt(m_i·P/β)·ξ, atom i outermost, then bead
/// j, then d; to normal modes with the RP-4 matrix; mode k scaled by √s_k; back to beads.
pub fn ref_redraw_dyn(masses: &[f64], p: usize, beta: f64, s: &[f64], rng: &mut Rng) -> Beads {
    let mut z = Beads::zeros(p, masses.len());
    for (i, &m) in masses.iter().enumerate() {
        let sd = (m * p as f64 / beta).sqrt();
        for j in 0..p {
            z.set(
                j,
                i,
                [sd * rng.normal(), sd * rng.normal(), sd * rng.normal()],
            );
        }
    }
    let mat = ref_matrix(p);
    let mut zt = to_modes(&mat, &z);
    for k in 0..p {
        for i in 0..masses.len() {
            zt.set(k, i, zt.get(k, i).map(|x| x * s[k].sqrt()));
        }
    }
    to_beads(&mat, &zt)
}

/// The 8-atom LJ cluster of `m7_common` (P beads, thermal momenta from `seed`) with thermostat
/// `th`, transform `transform` and adiabatic frequency `omega`.
pub fn pacmd_cluster_sim(
    p: usize,
    seed: u64,
    th: Thermostat,
    transform: pimd_core::Transform,
    omega: f64,
) -> RingPolymerSim {
    let ig = RingPolymerIntegrator::with_transform(CLUSTER_DT, CLUSTER_BETA, p, th, transform)
        .unwrap()
        .with_adiabatic_frequency(omega)
        .unwrap();
    RingPolymerSim::new(cluster_state(p, seed), None, ig, Box::new(lj())).unwrap()
}

/// 64-bit FNV-1a over little-endian bytes.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a of the bit patterns of the sim's positions, momenta, forces and energies, followed by
/// the next normal of the RNG (a fingerprint of the trajectory and of the RNG position).
pub fn fingerprint(sim: &RingPolymerSim, rng: &Rng) -> u64 {
    let mut bytes = Vec::new();
    let st = sim.state();
    for v in
        st.q.as_slice()
            .iter()
            .chain(st.p.as_slice())
            .chain(sim.forces().as_slice())
            .chain(sim.energies())
    {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    let mut r = Rng::restore(&rng.snapshot());
    bytes.extend_from_slice(&r.normal().to_bits().to_le_bytes());
    fnv1a(&bytes)
}
