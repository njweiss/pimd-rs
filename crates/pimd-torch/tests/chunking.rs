//! Bead-chunked evaluation (MILESTONE_8 "M8a: Required tests" 1, 2 and 4; SPEC P-6 as amended in
//! M8a, D-3 chunk-size sentence).
//!
//! P-6: one `evaluate` call evaluates the P beads in consecutive chunks of c beads (default c = P;
//! the last chunk holds the remaining P mod c beads), chunks in ascending bead order; each chunk is
//! one batched graph in which its b-th bead holds atoms bN … bN+N−1; energies and forces are stored
//! by bead index. So chunk k (beads kc … kc+b−1, b = min(c, P − kc)) is exactly the graph that an
//! unchunked potential builds for those b beads alone, and its results are bitwise those of
//! evaluating that chunk's beads as a P = b `Beads` (same edges, same graph, same libtorch thread
//! count; on CPU every call with a fixed thread count is bitwise reproducible, D-3, M6
//! `determinism.rs`). For c ≥ P there is one chunk: bitwise the unchunked result.
//!
//! D-3: results for different c agree only to stated tolerances: 1e-12 relative per bead (the
//! brief; the M6 batched vs per-bead tolerance, measured there ≤ 5e-15).
//!
//! Systems: the first five beads of the P = 8 reference systems of `data/mace_ref.json` (water and
//! ethanol in the gas phase, the 4-water box with boundary-crossing edges), the tiny model, CPU,
//! float64 (and float32 for the bitwise chunk rows), 1 libtorch thread.
#![cfg(feature = "torch")]

mod common;

use common::{SYSTEMS, bead, bitwise_eq, eval, export_path, load, max_abs, system};
use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, Potential, PotentialError};
use pimd_torch::{Device, Dtype, MacePotential};

/// P of the brief.
const P: usize = 5;
/// Chunk sizes of the brief: 1, uneven splits (2: 2+2+1, 3: 3+2), c = P and c > P.
const CHUNKS: [usize; 5] = [1, 2, 3, 5, 7];

/// Beads `s … s+b−1` of `q` as a b-bead `Beads`.
fn beads_range(q: &Beads, s: usize, b: usize) -> Beads {
    let n3 = q.n_atoms() * 3;
    Beads::from_vec(b, q.n_atoms(), q.as_slice()[s * n3..(s + b) * n3].to_vec()).unwrap()
}

/// The chunks of the brief for P beads and chunk size c: (first bead, beads in the chunk).
fn chunks_of(p: usize, c: usize) -> Vec<(usize, usize)> {
    (0..p).step_by(c).map(|s| (s, c.min(p - s))).collect()
}

/// A tiny-model potential with chunk size `c`.
fn chunked(z: &[u8], device: Device, dtype: Dtype, c: usize) -> MacePotential {
    load(z, device, dtype, 1)
        .with_beads_per_graph(c)
        .unwrap_or_else(|e| panic!("with_beads_per_graph({c}) failed: {e:?}"))
}

/// The P = 5 test systems: (name, atomic numbers, positions, cell).
fn systems() -> Vec<(String, Vec<u8>, Beads, Option<Cell>)> {
    SYSTEMS
        .iter()
        .map(|name| {
            let s = system(name);
            (
                name.to_string(),
                s.z.clone(),
                beads_range(&s.q, 0, P),
                s.cell,
            )
        })
        .collect()
}

/// Bitwise equality of bead `j` of `a` with bead `k` of `b` (energy and forces).
fn bead_bitwise(a: &Evaluation, j: usize, b: &Evaluation, k: usize) -> bool {
    a.energies[j].to_bits() == b.energies[k].to_bits()
        && bead(&a.forces, j)
            .iter()
            .zip(bead(&b.forces, k))
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Worst per-bead relative difference of `a` from `b`: (max_j |ΔE_j|/|E_j|, max_j
/// max|ΔF_j|/max|F_j|), with the values of `b` in the denominators.
fn rel_diff(a: &Evaluation, b: &Evaluation) -> (f64, f64) {
    let (mut er, mut fr) = (0.0f64, 0.0f64);
    for j in 0..b.energies.len() {
        er = er.max((a.energies[j] - b.energies[j]).abs() / b.energies[j].abs());
        let fb = bead(&b.forces, j);
        let df = max_abs(bead(&a.forces, j).iter().zip(fb).map(|(x, y)| x - y));
        fr = fr.max(df / max_abs(fb.iter().copied()));
    }
    (er, fr)
}

/// The shape checks of P-1 / S-4: P energies, forces `[P, N, 3]`, no virial (P-5).
fn assert_shape(e: &Evaluation, p: usize, n: usize, what: &str) {
    assert_eq!(e.energies.len(), p, "{what}: one energy per bead");
    assert_eq!(e.forces.n_beads(), p, "{what}: forces [P, N, 3]");
    assert_eq!(e.forces.n_atoms(), n, "{what}: forces [P, N, 3]");
    assert!(e.virial.is_none(), "{what}: virial is None (P-5)");
}

/// Chunk rows for one device and dtype: every chunk of every c bitwise equal to evaluating the
/// chunk's beads alone with an unchunked potential; c ≥ P bitwise equal to the unchunked result.
/// Returns the chunked results by c (for the cross-c comparison).
fn chunk_rows(device: Device, dtype: Dtype) -> Vec<(String, Evaluation, Vec<Evaluation>)> {
    let mut out = Vec::new();
    for (name, z, q, cell) in systems() {
        let n = q.n_atoms();
        let mut plain = load(&z, device, dtype, 1);
        let whole = eval(&mut plain, &q, cell.as_ref());
        assert_shape(&whole, P, n, &name);
        let mut by_c = Vec::new();
        for c in CHUNKS {
            let what = format!("{name}, {device:?} {dtype:?}, c = {c}");
            let mut pot = chunked(&z, device, dtype, c);
            assert_eq!(pot.beads_per_graph(), Some(c), "{what}: accessor");
            let got = eval(&mut pot, &q, cell.as_ref());
            assert_shape(&got, P, n, &what);
            for (s, b) in chunks_of(P, c) {
                let alone = eval(&mut plain, &beads_range(&q, s, b), cell.as_ref());
                assert_shape(&alone, b, n, &what);
                for k in 0..b {
                    assert!(
                        bead_bitwise(&got, s + k, &alone, k),
                        "{what}: bead {} (chunk at {s} of {b} beads) differs from evaluating \
                         that chunk alone (E {} vs {})",
                        s + k,
                        got.energies[s + k],
                        alone.energies[k]
                    );
                }
            }
            if c >= P {
                assert!(
                    bitwise_eq(&got, &whole),
                    "{what}: c >= P must be bitwise the unchunked result"
                );
            }
            by_c.push(got);
        }
        out.push((name, whole, by_c));
    }
    out
}

/// P-6, D-3 (MILESTONE_8 test 1): CPU float64, P = 5, c = 1, 2, 3, 5, 7, gas phase (water,
/// ethanol) and the 4-water box: each chunk's energies and forces are bitwise equal to evaluating
/// that chunk's beads alone as a P = b `Beads`; c ≥ P bitwise equals no chunking; results for
/// different c (and no chunking) agree to 1e-12 relative per bead.
#[test]
fn chunks_equal_separate_calls_f64() {
    for (name, whole, by_c) in chunk_rows(Device::Cpu, Dtype::F64) {
        for (c, got) in CHUNKS.iter().zip(&by_c) {
            let (er, fr) = rel_diff(got, &whole);
            eprintln!("{name}: c = {c} vs one graph |dE|/|E| {er:.2e}, |dF|/max|F| {fr:.2e}");
            assert!(er <= 1e-12, "{name}: c = {c}: energy {er:e}");
            assert!(fr <= 1e-12, "{name}: c = {c}: forces {fr:e}");
            for (c2, other) in CHUNKS.iter().zip(&by_c) {
                let (er, fr) = rel_diff(got, other);
                assert!(er <= 1e-12, "{name}: c = {c} vs c = {c2}: energy {er:e}");
                assert!(fr <= 1e-12, "{name}: c = {c} vs c = {c2}: forces {fr:e}");
            }
        }
    }
}

/// P-6, D-1 (MILESTONE_8 test 1, float32 row): the chunk rule holds bitwise with `Dtype::F32` on
/// the CPU as well (the cast happens per chunk on the way in and out, like a P = b call).
#[test]
fn chunks_equal_separate_calls_f32() {
    chunk_rows(Device::Cpu, Dtype::F32);
}

/// P-6, D-3 (MILESTONE_8 test 1, CUDA row): float32 on CUDA with `deterministic = true` (D-3: CUDA
/// results bitwise reproducible with the switch on): each chunk bitwise equals evaluating its
/// beads alone. Skipped with a message without CUDA.
#[test]
fn chunks_equal_separate_calls_cuda() {
    if !common::cuda_available() {
        eprintln!("SKIP chunks_equal_separate_calls_cuda: no CUDA device");
        return;
    }
    chunk_rows(Device::Cuda(0), Dtype::F32);
}

/// P-6 (MILESTONE_8 test 2): beads are independent and outputs are stored by bead index: for every
/// c and every bead k (including k = 4, the last, short chunk for c = 2 and c = 3, and k in the
/// middle of a chunk), displacing every atom of bead k (by up to 0.02 bohr) changes bead k's energy
/// and leaves every other bead's energy and forces unchanged to 1e-14 relative.
#[test]
fn moving_one_bead_changes_only_that_bead() {
    for (name, z, q, cell) in systems() {
        for c in CHUNKS {
            let mut pot = chunked(&z, Device::Cpu, Dtype::F64, c);
            let before = eval(&mut pot, &q, cell.as_ref());
            for k in 0..P {
                let mut moved = q.clone();
                let mut rng = common::SplitMix64(200 + k as u64);
                for i in 0..q.n_atoms() {
                    let x = moved.get(k, i);
                    moved.set(k, i, x.map(|v| v + rng.sym(0.02)));
                }
                let after = eval(&mut pot, &moved, cell.as_ref());
                for j in 0..P {
                    let de = (after.energies[j] - before.energies[j]).abs();
                    let fb = bead(&before.forces, j);
                    let df = max_abs(bead(&after.forces, j).iter().zip(fb).map(|(a, b)| a - b));
                    if j == k {
                        assert!(
                            de > 0.0,
                            "{name}, c = {c}: moving bead {k} must change its energy"
                        );
                    } else {
                        assert!(
                            de <= 1e-14 * before.energies[j].abs(),
                            "{name}, c = {c}: moving bead {k} changed bead {j}'s energy by {de:e}"
                        );
                        assert!(
                            df <= 1e-14 * max_abs(fb.iter().copied()),
                            "{name}, c = {c}: moving bead {k} changed bead {j}'s forces by {df:e}"
                        );
                    }
                }
            }
        }
    }
}

/// P-6 (MILESTONE_8 test 2, bead order): the chunked result of a bead-permuted ring polymer is the
/// same permutation of the chunked result (to 1e-12 relative, D-3: graph sizes differ between the
/// two calls), for c = 2 and 3: outputs are stored by bead index, not by position in a chunk.
#[test]
fn permuting_beads_permutes_results() {
    let perm = [4, 2, 0, 3, 1];
    for (name, z, q, cell) in systems() {
        let mut qp = q.clone();
        for (j, &src) in perm.iter().enumerate() {
            for i in 0..q.n_atoms() {
                qp.set(j, i, q.get(src, i));
            }
        }
        for c in [2, 3] {
            let mut pot = chunked(&z, Device::Cpu, Dtype::F64, c);
            let a = eval(&mut pot, &q, cell.as_ref());
            let b = eval(&mut pot, &qp, cell.as_ref());
            for (j, &src) in perm.iter().enumerate() {
                let de = (b.energies[j] - a.energies[src]).abs();
                assert!(
                    de <= 1e-12 * a.energies[src].abs(),
                    "{name}, c = {c}: bead {j} (was {src}) energy off by {de:e}"
                );
                let fa = bead(&a.forces, src);
                let df = max_abs(bead(&b.forces, j).iter().zip(fa).map(|(x, y)| x - y));
                assert!(
                    df <= 1e-12 * max_abs(fa.iter().copied()),
                    "{name}, c = {c}: bead {j} (was {src}) forces off by {df:e}"
                );
            }
        }
    }
}

/// P-6 public API (MILESTONE_8 test 4): `beads_per_graph()` is `None` after `load` (one graph of all
/// beads) and `Some(c)` after `with_beads_per_graph(c)`; c = 0 → `InvalidParameter { what: "beads
/// per graph", value: 0.0 }`.
#[test]
fn beads_per_graph_accessor_and_zero() {
    let pot = load(&[8, 1, 1], Device::Cpu, Dtype::F64, 1);
    assert_eq!(pot.beads_per_graph(), None);
    for c in [1, 2, 8, 1000] {
        let pot = load(&[8, 1, 1], Device::Cpu, Dtype::F64, 1)
            .with_beads_per_graph(c)
            .unwrap();
        assert_eq!(pot.beads_per_graph(), Some(c));
    }
    let pot =
        MacePotential::load(&export_path(), &[8, 1, 1], Device::Cpu, Dtype::F64, 1, true).unwrap();
    match pot.with_beads_per_graph(0) {
        Err(e) => assert_eq!(
            e,
            PotentialError::InvalidParameter {
                what: "beads per graph",
                value: 0.0
            }
        ),
        Ok(_) => panic!("c = 0 accepted"),
    }
}

/// P-6, P-4, P-1 (MILESTONE_8 test 4): the existing errors are unchanged with chunking (c = 2):
/// `subset = Some` → `Unsupported`; wrong atom count → `ShapeMismatch { expected: 3, actual }`; a
/// cell with r_max ≥ L_min/2 (L = 2·r_max exactly, and below) → `InvalidParameter { what:
/// "cutoff", value: r_max }`; and the potential still evaluates afterwards.
#[test]
fn existing_errors_unchanged_with_chunking() {
    let s = system("water");
    let q = beads_range(&s.q, 0, P);
    let mut pot = chunked(&s.z, Device::Cpu, Dtype::F64, 2);
    for sub in [&[][..], &[0][..], &[0, 1, 2][..]] {
        assert_eq!(
            pot.evaluate(&q, None, Some(sub)),
            Err(PotentialError::Unsupported),
            "subset {sub:?}"
        );
    }
    for n in [2, 4] {
        let mut bad = Beads::zeros(P, n);
        for j in 0..P {
            for i in 0..n {
                bad.set(j, i, [2.0 * i as f64, 0.1 * j as f64, 0.0]);
            }
        }
        assert_eq!(
            pot.evaluate(&bad, None, None),
            Err(PotentialError::ShapeMismatch {
                expected: 3,
                actual: n
            })
        );
    }
    let r_max = pot.cutoff();
    for l in [2.0 * r_max, 1.5 * r_max] {
        let cell = Cell::new([l + 5.0, l, l + 1.0]).unwrap();
        assert_eq!(
            pot.evaluate(&q, Some(&cell), None),
            Err(PotentialError::InvalidParameter {
                what: "cutoff",
                value: r_max
            }),
            "L_min = {l} bohr"
        );
    }
    let ok = eval(&mut pot, &q, None);
    assert_shape(&ok, P, 3, "after the errors");
}

/// P-6 (MILESTONE_8 test 3, part): with c = 3, changing P between calls (chunk layouts 3+2, then 2,
/// then 3+3+1, then 1, then 3+2 again) gives each call bitwise the result of a freshly loaded
/// potential with the same c for that P (no state kept from a call with another P or chunk size).
#[test]
fn changing_p_between_calls() {
    let s = system("waterbox");
    let mut pot = chunked(&s.z, Device::Cpu, Dtype::F64, 3);
    for p in [5, 2, 7, 1, 5, 8] {
        let q = beads_range(&s.q, 8 - p, p);
        let got = eval(&mut pot, &q, s.cell.as_ref());
        let mut fresh = chunked(&s.z, Device::Cpu, Dtype::F64, 3);
        let want = eval(&mut fresh, &q, s.cell.as_ref());
        assert_shape(&got, p, s.n, &format!("P = {p}"));
        assert!(
            bitwise_eq(&got, &want),
            "P = {p} after other bead counts differs from a fresh potential"
        );
        // And each chunk is the chunk alone (P-6).
        let mut plain = load(&s.z, Device::Cpu, Dtype::F64, 1);
        for (st, b) in chunks_of(p, 3) {
            let alone = eval(&mut plain, &beads_range(&q, st, b), s.cell.as_ref());
            for k in 0..b {
                assert!(
                    bead_bitwise(&got, st + k, &alone, k),
                    "P = {p}: bead {} differs from its chunk alone",
                    st + k
                );
            }
        }
    }
}
