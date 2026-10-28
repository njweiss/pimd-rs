//! Test-only helpers shared by the `pimd-integrators` integration tests.
//!
//! Everything here is built from the SPEC formulas directly (RP-4 matrix and frequencies, I-2
//! per-mode propagator, T-1 O step, B kicks with hand-written forces), independently of the code
//! under test. Only `pimd_core::Beads` (storage) and `pimd_core::Rng` (merged M1 code, D-2) are
//! used from the library.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums
#![allow(dead_code)]

use std::f64::consts::PI;

use pimd_core::{Beads, Rng};

/// SplitMix64 (Steele, Lea, Flood 2014). Seeded, deterministic, test-only.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Beads with every element uniform in [lo, hi).
pub fn random_beads(rng: &mut SplitMix64, p: usize, n: usize, lo: f64, hi: f64) -> Beads {
    let data = (0..p * n * 3).map(|_| rng.range(lo, hi)).collect();
    Beads::from_vec(p, n, data).unwrap()
}

/// C from RP-4, row-major P × P (C[j][k] at j*P + k).
pub fn ref_matrix(p: usize) -> Vec<f64> {
    let pf = p as f64;
    let mut c = vec![0.0; p * p];
    for j in 0..p {
        let jf = j as f64;
        c[j * p] = 1.0 / pf.sqrt();
        for m in 1..=(p - 1) / 2 {
            let arg = 2.0 * PI * (m as f64) * jf / pf;
            c[j * p + 2 * m - 1] = (2.0 / pf).sqrt() * arg.cos();
            c[j * p + 2 * m] = (2.0 / pf).sqrt() * arg.sin();
        }
        if p.is_multiple_of(2) {
            c[j * p + p - 1] = if j.is_multiple_of(2) { 1.0 } else { -1.0 } / pf.sqrt();
        }
    }
    c
}

/// ω_k from RP-2/RP-4, aligned with the columns of `ref_matrix`: ω_P = P/β, ω_0 = 0,
/// ω_{2m−1} = ω_{2m} = 2 ω_P sin(mπ/P), and ω_{P−1} = 2 ω_P for even P.
pub fn ref_freqs(p: usize, beta: f64) -> Vec<f64> {
    let pf = p as f64;
    let wp = pf / beta;
    let mut w = vec![0.0; p];
    for m in 1..=(p - 1) / 2 {
        let v = 2.0 * wp * ((m as f64) * PI / pf).sin();
        w[2 * m - 1] = v;
        w[2 * m] = v;
    }
    if p.is_multiple_of(2) {
        w[p - 1] = 2.0 * wp;
    }
    w
}

/// Bead → mode: x̃_k = Σ_j C_jk x_j (RP-4).
pub fn to_modes(c: &[f64], x: &Beads) -> Beads {
    let (p, n) = (x.n_beads(), x.n_atoms());
    let mut out = Beads::zeros(p, n);
    for k in 0..p {
        for i in 0..n {
            let mut v = [0.0; 3];
            for j in 0..p {
                let xj = x.get(j, i);
                for d in 0..3 {
                    v[d] += c[j * p + k] * xj[d];
                }
            }
            out.set(k, i, v);
        }
    }
    out
}

/// Mode → bead: x_j = Σ_k C_jk x̃_k (C orthogonal).
pub fn to_beads(c: &[f64], xt: &Beads) -> Beads {
    let (p, n) = (xt.n_beads(), xt.n_atoms());
    let mut out = Beads::zeros(p, n);
    for j in 0..p {
        for i in 0..n {
            let mut v = [0.0; 3];
            for k in 0..p {
                let xk = xt.get(k, i);
                for d in 0..3 {
                    v[d] += c[j * p + k] * xk[d];
                }
            }
            out.set(j, i, v);
        }
    }
    out
}

/// I-2 free propagation for time t, written per mode from the SPEC formula:
/// ω > 0: q̃' = cos(ωt) q̃ + sin(ωt)/(mω) p̃, p̃' = −mω sin(ωt) q̃ + cos(ωt) p̃; ω = 0: q̃' = q̃ + t p̃/m.
pub fn ref_free_propagate(
    q: &Beads,
    p: &Beads,
    masses: &[f64],
    beta: f64,
    t: f64,
) -> (Beads, Beads) {
    let nb = q.n_beads();
    let c = ref_matrix(nb);
    let w = ref_freqs(nb, beta);
    let mut qt = to_modes(&c, q);
    let mut pt = to_modes(&c, p);
    for k in 0..nb {
        for (i, &m) in masses.iter().enumerate() {
            let (mut a, mut b) = (qt.get(k, i), pt.get(k, i));
            for d in 0..3 {
                let (q0, p0) = (a[d], b[d]);
                if w[k] > 0.0 {
                    let (s, co) = (w[k] * t).sin_cos();
                    a[d] = co * q0 + s / (m * w[k]) * p0;
                    b[d] = -m * w[k] * s * q0 + co * p0;
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

/// c_k of T-1/T-2: c_k = exp(−γ_k dt) with γ_0 = 1/τ_0 (or 0 for `None`) and γ_k = 2 ω_k.
pub fn ref_c(p: usize, beta: f64, dt: f64, tau0: Option<f64>) -> Vec<f64> {
    let w = ref_freqs(p, beta);
    (0..p)
        .map(|k| {
            let g = if k == 0 {
                tau0.map_or(0.0, |t| 1.0 / t)
            } else {
                2.0 * w[k]
            };
            (-g * dt).exp()
        })
        .collect()
}

/// T-1 PILE-L O step: p̃_k ← c_k p̃_k + sqrt((1 − c_k²) m_i P/β) ξ, drawing ξ from `rng` in the
/// order k (outer), atom i, dimension d.
pub fn ref_o_step(
    p: &Beads,
    masses: &[f64],
    beta: f64,
    dt: f64,
    tau0: Option<f64>,
    rng: &mut Rng,
) -> Beads {
    let nb = p.n_beads();
    let pf = nb as f64;
    let c = ref_matrix(nb);
    let ck = ref_c(nb, beta, dt, tau0);
    let mut pt = to_modes(&c, p);
    for k in 0..nb {
        for (i, &m) in masses.iter().enumerate() {
            let mut v = pt.get(k, i);
            for x in &mut v {
                let xi = rng.normal();
                *x = ck[k] * *x + ((1.0 - ck[k] * ck[k]) * m * pf / beta).sqrt() * xi;
            }
            pt.set(k, i, v);
        }
    }
    to_beads(&c, &pt)
}

/// Harmonic (P-2) energies and forces written out: V_j = Σ_i ½ m_i ω² |q_{j,i}|², F = −m_i ω² q.
pub fn harmonic_ef(q: &Beads, masses: &[f64], omega: f64) -> (Vec<f64>, Beads) {
    let (p, n) = (q.n_beads(), q.n_atoms());
    let mut e = vec![0.0; p];
    let mut f = Beads::zeros(p, n);
    for j in 0..p {
        for (i, &m) in masses.iter().enumerate() {
            let x = q.get(j, i);
            let k = m * omega * omega;
            e[j] += 0.5 * k * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            f.set(j, i, x.map(|v| -k * v));
        }
    }
    (e, f)
}

/// DoubleWell (P-2) energies and forces written out: V = Σ a x⁴ − b x², F = −(4a x³ − 2b x).
pub fn double_well_ef(q: &Beads, a: f64, b: f64) -> (Vec<f64>, Beads) {
    let (p, n) = (q.n_beads(), q.n_atoms());
    let mut e = vec![0.0; p];
    let mut f = Beads::zeros(p, n);
    for j in 0..p {
        for i in 0..n {
            let x = q.get(j, i);
            for v in x {
                e[j] += a * v.powi(4) - b * v * v;
            }
            f.set(j, i, x.map(|v| -(4.0 * a * v.powi(3) - 2.0 * b * v)));
        }
    }
    (e, f)
}

/// p + h F, element-wise.
pub fn kick(p: &Beads, f: &Beads, h: f64) -> Beads {
    let data = p
        .as_slice()
        .iter()
        .zip(f.as_slice())
        .map(|(a, b)| a + h * b)
        .collect();
    Beads::from_vec(p.n_beads(), p.n_atoms(), data).unwrap()
}

/// H_P of RP-3: Σ p²/(2m) + Σ_j ½ m ω_P² |q_j − q_{j+1}|² (cyclic) + Σ_j V_j, ω_P = P/β.
pub fn ring_hamiltonian(q: &Beads, p: &Beads, masses: &[f64], beta: f64, v: &[f64]) -> f64 {
    let nb = q.n_beads();
    let wp = nb as f64 / beta;
    let mut h: f64 = v.iter().sum();
    for j in 0..nb {
        for (i, &m) in masses.iter().enumerate() {
            let pj = p.get(j, i);
            let (a, b) = (q.get(j, i), q.get((j + 1) % nb, i));
            for d in 0..3 {
                h += pj[d] * pj[d] / (2.0 * m) + 0.5 * m * wp * wp * (a[d] - b[d]).powi(2);
            }
        }
    }
    h
}

/// Largest |a − b| / max(|b|, floor) over all elements.
pub fn max_rel_diff(a: &Beads, b: &Beads, floor: f64) -> f64 {
    assert_eq!(a.n_beads(), b.n_beads());
    assert_eq!(a.n_atoms(), b.n_atoms());
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y).abs() / y.abs().max(floor))
        .fold(0.0, f64::max)
}

/// Momenta drawn from the RP-3 equilibrium N(0, m_i P/β) per component, from `rng`.
pub fn thermal_momenta(rng: &mut Rng, p: usize, masses: &[f64], beta: f64) -> Beads {
    let mut out = Beads::zeros(p, masses.len());
    for j in 0..p {
        for (i, &m) in masses.iter().enumerate() {
            let s = (m * p as f64 / beta).sqrt();
            out.set(j, i, [s * rng.normal(), s * rng.normal(), s * rng.normal()]);
        }
    }
    out
}
