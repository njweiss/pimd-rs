//! Tests for `pimd_core::NeighborList` (SPEC N-1, N-2, C-2, C-4, C-5).
//!
//! The expected lists come from the test's own brute-force search: every j ≠ i with
//! |min_image(r_j − r_i)| < cutoff + skin, min image written here as d − L·round(d/L) (C-2; ties at
//! exactly ±L/2 do not change |d|), j ascending. Configurations are drawn (from a seeded
//! SplitMix64 stream) until no pair distance lies within 1e-9 relative of cutoff + skin, so
//! membership is unambiguous and the comparison can be exact.

mod common;

use common::SplitMix64;
use pimd_core::{Cell, CoreError, NeighborList};

fn min_image(d: [f64; 3], l: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| d[k] - l[k] * (d[k] / l[k]).round())
}

fn dist2(a: [f64; 3], b: [f64; 3], l: Option<[f64; 3]>) -> f64 {
    let mut d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    if let Some(l) = l {
        d = min_image(d, l);
    }
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Brute-force full lists: neighbors of i = every j ≠ i with r_ij < rl, ascending.
fn brute(x: &[[f64; 3]], l: Option<[f64; 3]>, rl: f64) -> Vec<Vec<usize>> {
    (0..x.len())
        .map(|i| {
            (0..x.len())
                .filter(|&j| j != i && dist2(x[i], x[j], l) < rl * rl)
                .collect()
        })
        .collect()
}

/// Smallest | r_ij − rl | / rl over all pairs.
fn min_rel_gap(x: &[[f64; 3]], l: Option<[f64; 3]>, rl: f64) -> f64 {
    let mut g = f64::INFINITY;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            g = g.min((dist2(x[i], x[j], l).sqrt() - rl).abs() / rl);
        }
    }
    g
}

fn assert_matches_brute(
    nl: &NeighborList,
    x: &[[f64; 3]],
    l: Option<[f64; 3]>,
    rl: f64,
    what: &str,
) {
    assert_eq!(nl.n_atoms(), x.len(), "{what}: n_atoms");
    let want = brute(x, l, rl);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(nl.neighbors(i), &w[..], "{what}: neighbors({i})");
    }
}

/// Uniform points in [lo, hi) per component, redrawn until no pair is within 1e-9 relative of rl.
fn config(
    rng: &mut SplitMix64,
    n: usize,
    lo: [f64; 3],
    hi: [f64; 3],
    l: Option<[f64; 3]>,
    rl: f64,
) -> Vec<[f64; 3]> {
    loop {
        let x: Vec<[f64; 3]> = (0..n)
            .map(|_| [0, 1, 2].map(|k| rng.range(lo[k], hi[k])))
            .collect();
        if min_rel_gap(&x, l, rl) > 1e-9 {
            return x;
        }
    }
}

/// N-1: open boundaries (cell `None`), seeded random configurations of N = 1, 2, 50 and 300 atoms
/// spread over [−5, 15)³ (cutoff 2, skin 0.5) and N = 300 in [0, 6)³ (dense: about 70 neighbors
/// each), equal the brute-force lists exactly (membership and ascending order, no self).
#[test]
fn open_boundaries_match_brute_force() {
    let mut rng = SplitMix64::new(0x4E01);
    let (cutoff, skin) = (2.0, 0.5);
    for (n, lo, hi) in [
        (1, -5.0, 15.0),
        (2, -1.0, 1.0),
        (50, -5.0, 15.0),
        (300, -5.0, 15.0),
        (300, 0.0, 6.0),
    ] {
        let x = config(&mut rng, n, [lo; 3], [hi; 3], None, cutoff + skin);
        let mut nl = NeighborList::new(cutoff, skin).unwrap();
        nl.build(&x, None).unwrap();
        assert_matches_brute(&nl, &x, None, cutoff + skin, &format!("open N = {n}"));
    }
}

/// N-1, C-2, C-5: periodic, non-cubic cell L = (10, 12, 9) with cutoff + skin = 3.5 + 1.0 =
/// L_min/2 exactly (allowed by C-4). Seeded random configurations of N = 2, 60 and 300 atoms:
/// (a) inside the box, (b) the same points in unwrapped coordinates (each atom shifted by random
/// multiples of L in −2..2 per component), (c) with a third of the atoms within 1e-3 of a face
/// (x near 0 or near L). All equal the brute-force minimum-image lists exactly.
#[test]
fn periodic_matches_brute_force() {
    let l = [10.0, 12.0, 9.0];
    let cell = Cell::new(l).unwrap();
    let (cutoff, skin) = (3.5, 1.0);
    let rl = cutoff + skin;
    let mut rng = SplitMix64::new(0x4E02);
    for n in [2, 60, 300] {
        let x = config(&mut rng, n, [0.0; 3], l, Some(l), rl);
        let unwrapped: Vec<[f64; 3]> = x
            .iter()
            .map(|r| [0, 1, 2].map(|k| r[k] + l[k] * (rng.range(-2.0, 3.0).floor())))
            .collect();
        let mut faces = x.clone();
        for (i, r) in faces.iter_mut().enumerate() {
            if i % 3 == 0 {
                let k = i % 2;
                r[k] = if i % 6 == 0 {
                    rng.range(0.0, 1e-3)
                } else {
                    l[k] - rng.range(1e-12, 1e-3)
                };
            }
        }
        assert!(
            min_rel_gap(&faces, Some(l), rl) > 1e-9,
            "face configuration ambiguous"
        );
        for (what, pos) in [("box", &x), ("unwrapped", &unwrapped), ("faces", &faces)] {
            let mut nl = NeighborList::new(cutoff, skin).unwrap();
            nl.build(pos, Some(&cell)).unwrap();
            assert_matches_brute(&nl, pos, Some(l), rl, &format!("periodic {what} N = {n}"));
        }
    }
}

/// N-1, C-2: pairs near and beyond half the box along one axis. Cell L = 10 (cubic),
/// cutoff + skin = 4 + 1 = 5 = L/2. Relative to atom 0 at a = (0.2, 0.3, 0.4):
///   1: a + (4.99, 0, 0)    direct distance 4.99 < 5: neighbor;
///   2: a + (5.01, 0, 0)    direct 5.01 > L/2, minimum image −4.99: neighbor;
///   3: a + (0, −5.3, 0)    minimum image +4.7: neighbor;
///   4: a + (0, 0, 6.2)     minimum image −3.8: neighbor;
///   5: a + (3.6, 3.6, 0)   |d| = 5.09: not a neighbor;
///   6: a + (10, 0, 0)      the same point one box over, minimum image 0: neighbor (j ≠ i is about
///                          indices, not distance);
///   7: a + (−15.2, 0, 0)   minimum image −5.2 + 10 = 4.8: neighbor.
/// Also the whole list against the brute force.
#[test]
fn half_box_separations() {
    let cell = Cell::new([10.0; 3]).unwrap();
    let a = [0.2, 0.3, 0.4];
    let offs = [
        [0.0, 0.0, 0.0],
        [4.99, 0.0, 0.0],
        [5.01, 0.0, 0.0],
        [0.0, -5.3, 0.0],
        [0.0, 0.0, 6.2],
        [3.6, 3.6, 0.0],
        [10.0, 0.0, 0.0],
        [-15.2, 0.0, 0.0],
    ];
    let x: Vec<[f64; 3]> = offs
        .iter()
        .map(|o| [a[0] + o[0], a[1] + o[1], a[2] + o[2]])
        .collect();
    let mut nl = NeighborList::new(4.0, 1.0).unwrap();
    nl.build(&x, Some(&cell)).unwrap();
    assert_eq!(nl.neighbors(0), &[1, 2, 3, 4, 6, 7]);
    assert!(nl.neighbors(6).contains(&0) && !nl.neighbors(6).contains(&6));
    assert!(!nl.neighbors(5).contains(&0));
    assert_matches_brute(&nl, &x, Some([10.0; 3]), 5.0, "half box");
}

/// N-1: lists are full (j ∈ neighbors(i) ⇔ i ∈ neighbors(j)), sorted strictly ascending and never
/// contain i, on a dense periodic configuration (N = 200, ≈ 50 neighbors each).
#[test]
fn lists_are_full_sorted_and_exclude_self() {
    let l = [8.0, 8.0, 8.0];
    let cell = Cell::new(l).unwrap();
    let mut rng = SplitMix64::new(0x4E03);
    let x = config(&mut rng, 200, [0.0; 3], l, Some(l), 3.5);
    let mut nl = NeighborList::new(3.0, 0.5).unwrap();
    nl.build(&x, Some(&cell)).unwrap();
    let mut total = 0;
    for i in 0..200 {
        let ns = nl.neighbors(i);
        total += ns.len();
        assert!(
            ns.windows(2).all(|w| w[0] < w[1]),
            "neighbors({i}) not ascending"
        );
        assert!(!ns.contains(&i));
        for &j in ns {
            assert!(
                nl.neighbors(j).contains(&i),
                "{j} ∈ N({i}) but {i} ∉ N({j})"
            );
        }
    }
    assert!(total > 200 * 30, "configuration not dense enough: {total}");
}

/// Atoms of the rebuild tests: 50 seeded random points in a cubic box L = 10, cutoff 3, skin 1
/// (skin/2 = 0.5), atom 7 placed at x = 9.9 (0.1 from the upper face) and atom 8 at x = 0.05.
fn rebuild_setup(periodic: bool) -> (NeighborList, Vec<[f64; 3]>, Option<Cell>) {
    let l = [10.0; 3];
    let mut rng = SplitMix64::new(0x4E04);
    let mut x = config(&mut rng, 50, [0.0; 3], l, Some(l), 4.0);
    x[7][0] = 9.9;
    x[8][0] = 0.05;
    let cell = periodic.then(|| Cell::new(l).unwrap());
    let mut nl = NeighborList::new(3.0, 1.0).unwrap();
    nl.build(&x, cell.as_ref()).unwrap();
    (nl, x, cell)
}

fn moved(x: &[[f64; 3]], i: usize, d: [f64; 3]) -> Vec<[f64; 3]> {
    let mut y = x.to_vec();
    for k in 0..3 {
        y[i][k] += d[k];
    }
    y
}

/// N-2: rebuild iff some atom moved more than skin/2 = 0.5 (Euclidean norm), for open and periodic
/// lists. Unchanged positions: false. Atom 0, 25 or 49 displaced along a fixed unit direction
/// u = (2, −1, 2)/3 by 0.5(1 − 1e-6): false; by 0.5(1 + 1e-6): true; by 0.75 (between skin/2 and
/// skin): true. Diagonal (0.3, 0.3, 0.3), |d| = 0.52 with every component < 0.5: true;
/// (0.28, 0.28, 0.28), |d| = 0.485: false.
#[test]
fn rebuild_threshold_is_half_skin_euclidean() {
    let u = [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];
    for periodic in [false, true] {
        let (nl, x, _) = rebuild_setup(periodic);
        assert!(!nl.needs_rebuild(&x), "unchanged positions");
        for i in [0, 25, 49] {
            for (s, want) in [
                (0.5 * (1.0 - 1e-6), false),
                (0.5 * (1.0 + 1e-6), true),
                (0.75, true),
            ] {
                let y = moved(&x, i, u.map(|c| s * c));
                assert_eq!(
                    nl.needs_rebuild(&y),
                    want,
                    "periodic {periodic}: atom {i} moved by {s} (skin/2 = 0.5)"
                );
            }
        }
        assert!(
            nl.needs_rebuild(&moved(&x, 3, [0.3; 3])),
            "|d| = 0.52 > 0.5"
        );
        assert!(
            !nl.needs_rebuild(&moved(&x, 3, [0.28; 3])),
            "|d| = 0.485 < 0.5"
        );
    }
}

/// N-2, C-5: displacements are measured in unwrapped coordinates, never wrapped or minimum-imaged.
/// Atom 7 at x = 9.9 moving +0.45 to 10.35 (outside the box, crossing the face x = L) and atom 8
/// at x = 0.05 moving −0.45 to −0.4: displacement 0.45 < 0.5, false. Atom 7 moved by exactly L
/// along x (same periodic image, but unwrapped displacement 10): true; by L + 0.1: true; atom 8 by
/// (0, −L, 0.3): true.
#[test]
fn rebuild_uses_unwrapped_displacements() {
    let (nl, x, cell) = rebuild_setup(true);
    assert!(cell.is_some());
    assert!(
        !nl.needs_rebuild(&moved(&x, 7, [0.45, 0.0, 0.0])),
        "crossing x = L by 0.45"
    );
    assert!(
        !nl.needs_rebuild(&moved(&x, 8, [-0.45, 0.0, 0.0])),
        "crossing x = 0 by 0.45"
    );
    let both = moved(&moved(&x, 7, [0.45, 0.0, 0.0]), 8, [-0.45, 0.0, 0.0]);
    assert!(!nl.needs_rebuild(&both));
    assert!(
        nl.needs_rebuild(&moved(&x, 7, [10.0, 0.0, 0.0])),
        "moved by L"
    );
    assert!(
        nl.needs_rebuild(&moved(&x, 7, [10.1, 0.0, 0.0])),
        "moved by L + 0.1"
    );
    assert!(
        nl.needs_rebuild(&moved(&x, 8, [0.0, -10.0, 0.3])),
        "moved by (0, −L, 0.3)"
    );
}

/// N-2: the reference positions are those of the last build. After rebuilding at displaced
/// positions, those positions need no rebuild, the original ones (now 0.75 away) do. With skin = 0
/// any non-zero displacement requires a rebuild and none does not.
#[test]
fn rebuild_reference_is_last_build() {
    let (mut nl, x, cell) = rebuild_setup(true);
    let y = moved(&x, 12, [0.75, 0.0, 0.0]);
    assert!(nl.needs_rebuild(&y));
    nl.build(&y, cell.as_ref()).unwrap();
    assert!(!nl.needs_rebuild(&y));
    assert!(nl.needs_rebuild(&x));
    let mut z = NeighborList::new(3.0, 0.0).unwrap();
    z.build(&x, cell.as_ref()).unwrap();
    assert!(!z.needs_rebuild(&x));
    assert!(z.needs_rebuild(&moved(&x, 0, [1e-9, 0.0, 0.0])));
}

/// N-1, N-2 (the Verlet guarantee): while no atom has moved more than skin/2 the old list still
/// contains every pair within the cutoff. Every atom of a dense periodic configuration (N = 200,
/// cell 8³, cutoff 2.5, skin 0.8) is moved by a random vector of length ≤ 0.49·skin (in unwrapped
/// coordinates, so some cross faces); `needs_rebuild` is false and every pair with
/// |min_image(r_j − r_i)| < cutoff at the new positions is in the old list.
#[test]
fn stale_list_still_covers_cutoff() {
    let l = [8.0; 3];
    let cell = Cell::new(l).unwrap();
    let (cutoff, skin) = (2.5, 0.8);
    let mut rng = SplitMix64::new(0x4E05);
    let x = config(&mut rng, 200, [0.0; 3], l, Some(l), cutoff + skin);
    let mut nl = NeighborList::new(cutoff, skin).unwrap();
    nl.build(&x, Some(&cell)).unwrap();
    let y: Vec<[f64; 3]> = x
        .iter()
        .map(|r| {
            let d = [0, 1, 2].map(|_| rng.range(-1.0, 1.0));
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let s = 0.49 * skin * rng.next_f64() / n;
            [r[0] + s * d[0], r[1] + s * d[1], r[2] + s * d[2]]
        })
        .collect();
    assert!(!nl.needs_rebuild(&y));
    let close = brute(&y, Some(l), cutoff);
    let mut pairs = 0;
    for (i, c) in close.iter().enumerate() {
        for &j in c {
            pairs += 1;
            assert!(
                nl.neighbors(i).contains(&j),
                "pair ({i}, {j}) within cutoff missing"
            );
        }
    }
    assert!(pairs > 1000);
}

/// C-4: `build` with a cell requires cutoff + skin ≤ L_min/2, else `CoreError::InvalidValue`
/// (the existing variant for an out-of-range value). cutoff + skin = 3.5 + 1 = 4.5: a cell with
/// one edge 8.9 (L/2 = 4.45) is rejected, for each axis; edge 9 (L/2 = 4.5) is accepted; open
/// boundaries accept it with any positions.
#[test]
fn build_checks_cell_size() {
    let x = vec![[0.0; 3], [1.0, 1.0, 1.0]];
    for axis in 0..3 {
        let mut l = [20.0; 3];
        l[axis] = 8.9;
        let mut nl = NeighborList::new(3.5, 1.0).unwrap();
        let r = nl.build(&x, Some(&Cell::new(l).unwrap()));
        assert!(
            matches!(r, Err(CoreError::InvalidValue { .. })),
            "axis {axis}: {r:?}"
        );
        l[axis] = 9.0;
        assert_eq!(
            nl.build(&x, Some(&Cell::new(l).unwrap())),
            Ok(()),
            "axis {axis}"
        );
    }
    let mut nl = NeighborList::new(3.5, 1.0).unwrap();
    assert_eq!(nl.build(&x, None), Ok(()));
}

/// N-1: `new` requires cutoff > 0 and finite, skin ≥ 0 and finite (`CoreError::InvalidValue`);
/// skin = 0 is allowed. `n_atoms` follows the last build.
#[test]
fn new_validates_and_n_atoms_follows_build() {
    for c in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(
                NeighborList::new(c, 0.5),
                Err(CoreError::InvalidValue { .. })
            ),
            "cutoff {c}"
        );
    }
    for s in [-1e-12, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(
                NeighborList::new(2.0, s),
                Err(CoreError::InvalidValue { .. })
            ),
            "skin {s}"
        );
    }
    let mut nl = NeighborList::new(2.0, 0.0).unwrap();
    let mut rng = SplitMix64::new(0x4E06);
    let x = config(&mut rng, 30, [0.0; 3], [5.0; 3], None, 2.0);
    nl.build(&x, None).unwrap();
    assert_eq!(nl.n_atoms(), 30);
    assert_matches_brute(&nl, &x, None, 2.0, "skin 0");
    nl.build(&x[..10], None).unwrap();
    assert_eq!(nl.n_atoms(), 10);
    assert_matches_brute(&nl, &x[..10], None, 2.0, "rebuilt with 10 atoms");
}
