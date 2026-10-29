//! Test-only helpers for the `pimd-potentials` M3 tests.
//!
//! Everything here is written from the SPEC formulas (P-3, P-5, E-8, C-2), independently of the
//! code under test: a seeded generator, random configurations, the test's own minimum image and an
//! O(N²) brute-force Lennard-Jones over all pairs i < j.
#![allow(dead_code)]

pub mod lammps;

use pimd_core::Beads;

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

    /// Uniform integer in [lo, hi].
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + ((hi - lo + 1) as f64 * self.next_f64()).floor() as i64
    }
}

/// The test's own minimum image (C-2): d − L·round(d/L), component-wise, with lengths `l`.
/// (Boundary ties at exactly ±L/2 do not matter here: such pairs are beyond every cutoff used.)
pub fn min_image(d: [f64; 3], l: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| d[k] - l[k] * (d[k] / l[k]).round())
}

/// Displacement r_i − r_j, minimum image when `l` is `Some`.
pub fn disp(a: [f64; 3], b: [f64; 3], l: Option<[f64; 3]>) -> [f64; 3] {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    match l {
        Some(l) => min_image(d, l),
        None => d,
    }
}

pub fn norm2(d: [f64; 3]) -> f64 {
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// `n` points in the box [0, L) by random sequential insertion with minimum-image separation at
/// least `dmin` (periodic when `periodic`, else plain distance), from `rng`.
pub fn random_config(
    rng: &mut SplitMix64,
    n: usize,
    l: [f64; 3],
    dmin: f64,
    periodic: bool,
) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(n);
    let pl = periodic.then_some(l);
    let mut tries = 0;
    while out.len() < n {
        tries += 1;
        assert!(tries < 1_000_000, "random_config: box too dense");
        let x = [0, 1, 2].map(|k| rng.range(0.0, l[k]));
        if out.iter().all(|y| norm2(disp(x, *y, pl)) >= dmin * dmin) {
            out.push(x);
        }
    }
    out
}

/// Shifts each atom by a random integer multiple (−2..=2) of L per component: the same periodic
/// system in unwrapped coordinates (C-5).
pub fn unwrap_randomly(rng: &mut SplitMix64, x: &[[f64; 3]], l: [f64; 3]) -> Vec<[f64; 3]> {
    x.iter()
        .map(|r| [0, 1, 2].map(|k| r[k] + l[k] * rng.int(-2, 2) as f64))
        .collect()
}

/// `[1, N, 3]` beads from N 3-vectors.
pub fn beads1(x: &[[f64; 3]]) -> Beads {
    beads(&[x.to_vec()])
}

/// `[P, N, 3]` beads from P configurations of N 3-vectors.
pub fn beads(conf: &[Vec<[f64; 3]>]) -> Beads {
    let n = conf[0].len();
    let mut b = Beads::zeros(conf.len(), n);
    for (j, c) in conf.iter().enumerate() {
        assert_eq!(c.len(), n);
        for (i, r) in c.iter().enumerate() {
            b.set(j, i, *r);
        }
    }
    b
}

/// One bead of `b` as N 3-vectors.
pub fn bead(b: &Beads, j: usize) -> Vec<[f64; 3]> {
    (0..b.n_atoms()).map(|i| b.get(j, i)).collect()
}

/// LJ pair energy 4ε[(σ/r)¹² − (σ/r)⁶] and F(r)/r with F(r) = −dV/dr = 24ε[2(σ/r)¹² − (σ/r)⁶]/r,
/// as functions of r² (P-3).
pub fn lj_pair(eps: f64, sig: f64, r2: f64) -> (f64, f64) {
    let s6 = (sig * sig / r2).powi(3);
    (
        4.0 * eps * (s6 * s6 - s6),
        24.0 * eps * (2.0 * s6 * s6 - s6) / r2,
    )
}

/// Brute-force LJ of one configuration over all pairs i < j (P-3, P-5, E-8): energy (shifted by
/// V(r_c) inside r_c when `shift`), forces −∇V, virial W = Σ_{i<j} r_ij ⊗ f_ij with
/// r_ij = r_i − r_j (minimum image when `l` is `Some`) and f_ij the force on i from j, and the
/// number of pairs with r < r_c.
pub struct BruteLj {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub virial: [[f64; 3]; 3],
    pub n_inside: usize,
}

pub fn brute_lj(
    x: &[[f64; 3]],
    l: Option<[f64; 3]>,
    eps: f64,
    sig: f64,
    rc: f64,
    shift: bool,
) -> BruteLj {
    let n = x.len();
    let vc = if shift {
        lj_pair(eps, sig, rc * rc).0
    } else {
        0.0
    };
    let mut out = BruteLj {
        energy: 0.0,
        forces: vec![[0.0; 3]; n],
        virial: [[0.0; 3]; 3],
        n_inside: 0,
    };
    for i in 0..n {
        for j in i + 1..n {
            let d = disp(x[i], x[j], l);
            let r2 = norm2(d);
            if r2 >= rc * rc {
                continue;
            }
            out.n_inside += 1;
            let (e, f_over_r) = lj_pair(eps, sig, r2);
            out.energy += e - vc;
            for a in 0..3 {
                out.forces[i][a] += f_over_r * d[a];
                out.forces[j][a] -= f_over_r * d[a];
                for b in 0..3 {
                    out.virial[a][b] += d[a] * f_over_r * d[b];
                }
            }
        }
    }
    out
}

/// Smallest |r − r_c| over all pairs (minimum image when `l` is `Some`).
pub fn min_gap_to_cutoff(x: &[[f64; 3]], l: Option<[f64; 3]>, rc: f64) -> f64 {
    let mut gap = f64::INFINITY;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            gap = gap.min((norm2(disp(x[i], x[j], l)).sqrt() - rc).abs());
        }
    }
    gap
}
