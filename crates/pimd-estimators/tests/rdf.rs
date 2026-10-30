//! Tests for `Rdf` (SPEC E-7, C-2, C-5): hand-counted examples, an ideal-gas statistical test and
//! LAMMPS `compute rdf` on the LJ snapshot. Distances in the test use its own minimum image,
//! d − L·round(d/L) (C-2).

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

use std::f64::consts::PI;

use lammps::{Snapshot, snapshot_json};
use pimd_core::{Beads, Cell};
use pimd_estimators::{EstimatorError, Rdf};

fn conf(p: &[Vec<[f64; 3]>]) -> Beads {
    let n = p[0].len();
    let mut b = Beads::zeros(p.len(), n);
    for (j, c) in p.iter().enumerate() {
        for (i, r) in c.iter().enumerate() {
            b.set(j, i, *r);
        }
    }
    b
}

fn assert_close_vec(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (b, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: bin {b}: {g} vs {w} (all: {got:?})"
        );
    }
}

/// E-7: hand-counted three-atom example, P = 1. Cell L = 10 (V = 1000), r_max = 5 = L/2, 5 bins of
/// width 1, bin centres 0.5 … 4.5.
/// Sample 1: a0 = (0.5, 0.5, 0.5), a1 = (2, 0.5, 0.5), a2 = (0.5, 0.5, 9.6):
///   r01 = 1.5 (bin 1); r02 = |9.1 − 10| = 0.9 through the minimum image (bin 0);
///   r12 = √(1.5² + 0.9²) = √3.06 ≈ 1.749 (bin 1). C = (1, 2, 0, 0, 0).
///   g_b = 2 C_b V / (P N(N−1) V_shell(b)) with N(N−1) = 6, V_shell(b) = (4π/3)((b+1)³ − b³):
///   g_0 = 2000 / (6 · 4π/3) = 250/π; g_1 = 4000 / (6 · 28π/3) = 500/(7π); others 0.
/// Sample 2: a0 = (0.5, 0.5, 0.5), a1 = (3, 0.5, 0.5), a2 = (5.4, 5.4, 0.5):
///   r01 = 2.5 (bin 2); r02 = √(4.9² + 4.9²) ≈ 6.93 > r_max; r12 = √(2.4² + 4.9²) ≈ 5.46 > r_max.
///   g_2 = 2000 / (6 · 76π/3) = 250/(19π).
/// After both samples g is the average: (125/π, 250/(7π), 125/(19π), 0, 0); n_samples = 2.
#[test]
fn three_atoms_hand_counted() {
    let cell = Cell::new([10.0; 3]).unwrap();
    let mut rdf = Rdf::new(5, 5.0, cell, 3).unwrap();
    assert_eq!(rdf.n_samples(), 0);
    assert_close_vec(
        &rdf.bin_centers(),
        &[0.5, 1.5, 2.5, 3.5, 4.5],
        1e-15,
        "centres",
    );
    let s1 = conf(&[vec![[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 0.5, 9.6]]]);
    rdf.accumulate(&s1).unwrap();
    assert_eq!(rdf.n_samples(), 1);
    assert_close_vec(
        &rdf.g(),
        &[250.0 / PI, 500.0 / (7.0 * PI), 0.0, 0.0, 0.0],
        1e-13,
        "sample 1",
    );
    let s2 = conf(&[vec![[0.5, 0.5, 0.5], [3.0, 0.5, 0.5], [5.4, 5.4, 0.5]]]);
    rdf.accumulate(&s2).unwrap();
    assert_eq!(rdf.n_samples(), 2);
    assert_close_vec(
        &rdf.g(),
        &[
            125.0 / PI,
            250.0 / (7.0 * PI),
            125.0 / (19.0 * PI),
            0.0,
            0.0,
        ],
        1e-13,
        "average of two samples",
    );
}

/// E-7: hand-counted two-atom example with P = 2 beads (pairs within each bead only; the bead
/// count enters as 1/P). Cell L = 10, r_max = 5, 5 bins. Bead 0: a0 = (1, 1, 1), a1 = (1, 4.5, 1),
/// r = 3.5 (bin 3). Bead 1: a0 = (1, 1, 1), a1 = (1, 1, −4.8): dz = −5.8, minimum image +4.2
/// (bin 4). (Cross-bead pairs would put a0(bead 0)–a1(bead 1), distance 4.2, and
/// a0(bead 1)–a1(bead 0), 3.5, in the same bins again and double the counts.) With N(N−1) = 2,
/// P = 2: g_3 = 2·1·1000 / (2·2·(4π/3)(64 − 27)) = 375/(37π); g_4 = 2000 / (4·(4π/3)·61) =
/// 375/(61π); others 0.
#[test]
fn two_atoms_two_beads_hand_counted() {
    let cell = Cell::new([10.0; 3]).unwrap();
    let mut rdf = Rdf::new(5, 5.0, cell, 2).unwrap();
    let q = conf(&[
        vec![[1.0, 1.0, 1.0], [1.0, 4.5, 1.0]],
        vec![[1.0, 1.0, 1.0], [1.0, 1.0, -4.8]],
    ]);
    rdf.accumulate(&q).unwrap();
    assert_eq!(rdf.n_samples(), 1);
    assert_close_vec(
        &rdf.g(),
        &[0.0, 0.0, 0.0, 375.0 / (37.0 * PI), 375.0 / (61.0 * PI)],
        1e-13,
        "P = 2",
    );
}

/// E-7, C-5: positions are unwrapped. Shifting atoms of the three-atom example by different
/// integer multiples of L (up to ±30 bohr) gives the same g (1e-13 relative).
#[test]
fn unwrapped_positions_give_same_g() {
    let cell = Cell::new([10.0; 3]).unwrap();
    let base = vec![[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 0.5, 9.6]];
    let shifted = vec![[20.5, -9.5, 0.5], [2.0, 30.5, -19.5], [-29.5, 0.5, -0.4]];
    let mut a = Rdf::new(5, 5.0, cell, 3).unwrap();
    let mut b = Rdf::new(5, 5.0, cell, 3).unwrap();
    a.accumulate(&conf(&[base])).unwrap();
    b.accumulate(&conf(&[shifted])).unwrap();
    assert_close_vec(&b.g(), &a.g(), 1e-13, "unwrapped");
}

/// SplitMix64, test-only seeded generator independent of `pimd_core::Rng`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// E-7: ideal gas. N = 64 independent uniform points in the cell (10, 11, 12), r_max = 5 = L_min/2,
/// 25 bins. For r_max ≤ L_min/2 the sphere of radius r_max lies inside the minimum-image cell, so
/// the minimum-image distance of two uniform points falls in shell b with probability exactly
/// V_shell(b)/V and E[g_b] = 1 for every bin.
/// 3 000 independent configurations in 30 blocks of 100 (one `Rdf` per block; configurations are
/// independent, so the blocks are). Per bin: |mean of block g − 1| ≤ 5σ, σ = std of the block
/// means / √30. With 25 bins the chance of any false failure at 5σ is about 1.5e-5 (Gaussian);
/// the innermost bin has about 0.05 expected pairs per configuration, 150 in total, so its σ is
/// about 0.08 and the test also catches a factor-2 error everywhere.
#[test]
fn ideal_gas_is_one() {
    let l = [10.0, 11.0, 12.0];
    let cell = Cell::new(l).unwrap();
    let (n, n_bins, blocks, per_block) = (64, 25, 30, 100);
    let mut rng = SplitMix64(0xE7);
    let mut block_g: Vec<Vec<f64>> = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        let mut rdf = Rdf::new(n_bins, 5.0, cell, n).unwrap();
        for _ in 0..per_block {
            let data = (0..n * 3).map(|k| l[k % 3] * rng.next_f64()).collect();
            rdf.accumulate(&Beads::from_vec(1, n, data).unwrap())
                .unwrap();
        }
        assert_eq!(rdf.n_samples(), per_block);
        block_g.push(rdf.g());
    }
    for b in 0..n_bins {
        let x: Vec<f64> = block_g.iter().map(|g| g[b]).collect();
        let m = x.iter().sum::<f64>() / blocks as f64;
        let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (blocks - 1) as f64;
        let sigma = (var / blocks as f64).sqrt();
        assert!(sigma > 0.0 && sigma < 0.2, "bin {b}: sigma {sigma}");
        assert!(
            (m - 1.0).abs() <= 5.0 * sigma,
            "bin {b}: g = {m} ± {sigma}, want 1"
        );
    }
}

/// E-7: LAMMPS `compute rdf 68 cutoff 3.4` on the 256-atom snapshot (cubic L = 6.8399, so
/// r_max = 3.4 ≤ L/2 = 3.42). LAMMPS's normalisation for one atom type is
/// g_b = hist_b / ((N − 1) N (4π/3V)(r_{b+1}³ − r_b³)) with every pair counted twice, i.e. exactly
/// E-7 with P = 1 (see `gen_lj_snapshot.py`; no convention difference). g matches to 1e-10
/// (relative, floor 1) and the bin centres to 1e-12. The generator checked that no pair distance
/// lies within 1e-9 of a bin edge (minimum 1.3e-6), so the binning is unambiguous.
#[test]
fn snapshot_matches_lammps() {
    let s = Snapshot::load();
    let json = snapshot_json();
    let r = json.get("rdf");
    let n_bins = r.get("n_bins").num() as usize;
    let r_max = r.get("r_max").num();
    let mut rdf = Rdf::new(n_bins, r_max, s.cell(), s.n).unwrap();
    rdf.accumulate(&s.q).unwrap();
    assert_close_vec(
        &rdf.bin_centers(),
        &r.get("bin_centers").nums(),
        1e-12,
        "centres",
    );
    assert_close_vec(&rdf.g(), &r.get("g").nums(), 1e-10, "g vs LAMMPS");
}

/// E-7: `new` requires n_bins ≥ 1 and 0 < r_max ≤ L_min/2 (finite), else
/// `EstimatorError::InvalidValue`; r_max = L_min/2 exactly is accepted. `accumulate` with the wrong
/// atom count gives `EstimatorError::ShapeMismatch`.
#[test]
fn errors() {
    let cell = Cell::new([10.0, 8.0, 12.0]).unwrap();
    assert!(matches!(
        Rdf::new(0, 2.0, cell, 4),
        Err(EstimatorError::InvalidValue { .. })
    ));
    for r in [0.0, -1.0, 4.0 + 1e-12, 4.5, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                Rdf::new(10, r, cell, 4),
                Err(EstimatorError::InvalidValue { .. })
            ),
            "r_max = {r} accepted (L_min/2 = 4)"
        );
    }
    let mut rdf = Rdf::new(10, 4.0, cell, 4).unwrap();
    for n in [3, 5] {
        assert!(
            matches!(
                rdf.accumulate(&Beads::zeros(1, n)),
                Err(EstimatorError::ShapeMismatch { .. })
            ),
            "{n} atoms"
        );
    }
    assert!(rdf.accumulate(&Beads::zeros(2, 4)).is_ok());
}

/// E-7 (Q10): N(N − 1) must be non-zero, so `Rdf::new` with 0 or 1 atoms is `InvalidValue`; 2
/// atoms are accepted. Before any `accumulate`, `g` is all zeros (length `n_bins`) and
/// `n_samples` is 0.
#[test]
fn too_few_atoms_and_no_samples() {
    let cell = Cell::new([10.0; 3]).unwrap();
    for n in [0, 1] {
        assert!(
            matches!(
                Rdf::new(5, 5.0, cell, n),
                Err(EstimatorError::InvalidValue { .. })
            ),
            "{n} atoms"
        );
    }
    let rdf = Rdf::new(5, 5.0, cell, 2).unwrap();
    assert_eq!(rdf.n_samples(), 0);
    assert_eq!(rdf.g(), vec![0.0; 5]);
}

/// E-7 (Q12): a configuration with no beads (P = 0) is `ShapeMismatch` and leaves the
/// accumulator unchanged: a later valid sample gives the same g as on a fresh `Rdf`.
#[test]
fn zero_beads_is_rejected() {
    let cell = Cell::new([10.0; 3]).unwrap();
    let mut rdf = Rdf::new(5, 5.0, cell, 2).unwrap();
    assert!(matches!(
        rdf.accumulate(&Beads::zeros(0, 2)),
        Err(EstimatorError::ShapeMismatch { .. })
    ));
    assert_eq!(rdf.n_samples(), 0);
    let q = conf(&[vec![[0.5; 3], [2.0, 0.5, 0.5]]]);
    rdf.accumulate(&q).unwrap();
    let mut fresh = Rdf::new(5, 5.0, cell, 2).unwrap();
    fresh.accumulate(&q).unwrap();
    assert_eq!(rdf.g(), fresh.g());
    assert_eq!(rdf.n_samples(), 1);
}

/// E-7 (Q14): `from_parts(.., counts(), n_samples())` restores an RDF exactly. Three hand
/// configurations (P = 1, then P = 2, then P = 1, N = 3, L = 10) are accumulated into `a`;
/// after the first two, `b` is rebuilt from `a`'s parts and both then accumulate the third. `g`,
/// `counts` and `n_samples` of `a` and `b` are bitwise equal at both points. `counts` has length
/// `n_bins` and is all zeros on a fresh RDF.
#[test]
fn from_parts_restores_exactly() {
    let cell = Cell::new([10.0; 3]).unwrap();
    let c1 = conf(&[vec![[0.5; 3], [2.0, 0.5, 0.5], [0.5, 0.5, 9.6]]]);
    let c2 = conf(&[
        vec![[0.5; 3], [3.0, 0.5, 0.5], [5.4, 5.4, 0.5]],
        vec![[1.0; 3], [1.7, 1.0, 1.0], [1.0, 4.3, 1.0]],
    ]);
    let c3 = conf(&[vec![[9.9, 0.1, 0.1], [0.3, 0.1, 0.1], [4.0, 4.0, 4.0]]]);
    let mut a = Rdf::new(5, 5.0, cell, 3).unwrap();
    assert_eq!(a.counts(), &[0.0; 5]);
    a.accumulate(&c1).unwrap();
    a.accumulate(&c2).unwrap();
    let mut b = Rdf::from_parts(5, 5.0, cell, 3, a.counts().to_vec(), a.n_samples()).unwrap();
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&b.g()), bits(&a.g()));
    assert_eq!(bits(b.counts()), bits(a.counts()));
    assert_eq!(b.n_samples(), 2);
    a.accumulate(&c3).unwrap();
    b.accumulate(&c3).unwrap();
    assert_eq!(bits(&b.g()), bits(&a.g()));
    assert_eq!(bits(b.counts()), bits(a.counts()));
    assert_eq!(b.n_samples(), 3);
    assert!(a.counts().iter().any(|&c| c > 0.0));
}

/// E-7 (Q14): `from_parts` validates like `new` (`InvalidValue` for 0 bins, r_max > L_min/2, fewer
/// than 2 atoms) and rejects `counts` of the wrong length with `ShapeMismatch`.
#[test]
fn from_parts_validates() {
    let cell = Cell::new([10.0, 8.0, 12.0]).unwrap();
    let ok = |n| vec![0.0; n];
    for r in [
        Rdf::from_parts(0, 2.0, cell, 4, ok(0), 0),
        Rdf::from_parts(4, 4.5, cell, 4, ok(4), 0),
        Rdf::from_parts(4, 2.0, cell, 1, ok(4), 0),
    ] {
        assert!(
            matches!(r, Err(EstimatorError::InvalidValue { .. })),
            "{r:?}"
        );
    }
    for n in [3, 5] {
        let r = Rdf::from_parts(4, 2.0, cell, 4, ok(n), 1);
        assert!(
            matches!(r, Err(EstimatorError::ShapeMismatch { .. })),
            "{n} counts: {r:?}"
        );
    }
    assert!(Rdf::from_parts(4, 4.0, cell, 4, ok(4), 0).is_ok());
}
