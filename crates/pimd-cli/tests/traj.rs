//! `pimd-traj` (MILESTONE_9 M9b): `trajectory.xyz` (MILESTONE_5 "Trajectory output") → extended
//! XYZ for OVITO, VMD and ASE. Required tests 1 (format), 2 (wrapping), 3 (restart joining),
//! 5 (errors) and 6 (end-to-end); test 4 (ASE) is `long_traj_ase.rs`. Details the brief leaves
//! open are pinned in QUESTIONS.md Q40.
//!
//! Wrapping (M9b "Wrapping", amended by the owner on Q40 item 3): molecules are the connected
//! components of the first kept frame's centroids, edges between atoms closer than
//! `--bond-angstrom` (strictly) under the C-2 minimum image. Make-whole, once, in that frame: a
//! breadth-first walk from each molecule's lowest-index atom (FIFO queue; from the atom taken off
//! the queue, its unvisited bonded atoms in ascending index) gives each newly reached atom j, from
//! parent i, the lattice offset n_j = n_i + k·L with k = −floor((c_j − c_i)/L + 1/2) per component
//! (c_j placed at the minimum image of the parent's placed centroid; n_root = 0). Every frame, each
//! atom's beads and centroid move by its fixed offset n_j, then every molecule's beads and
//! centroids by one lattice vector −floor(c/L)·L, c the (offset) centroid of its lowest-index atom
//! (C-3 interval [0, L)). `--no-wrap` or no box: neither step. Expected values below are
//! hand-derived; inputs are multiples of 0.25 Å so every expected number is exact in f64.

#![allow(clippy::needless_range_loop)]

mod traj_common;

use std::path::Path;

use traj_common::*;

/// Shorthand: a row of a trajectory frame.
fn r(sym: &str, x: f64, y: f64, z: f64) -> (&str, [f64; 3]) {
    (sym, [x, y, z])
}

/// Exact comparison of read-back output rows with expected (symbol, position, bead).
fn assert_rows(got: &XFrame, want: &[(&str, [f64; 3], Option<usize>)], what: &str) {
    assert_eq!(got.rows.len(), want.len(), "{what}: row count");
    for (k, (g, w)) in got.rows.iter().zip(want).enumerate() {
        assert_eq!(g.0, w.0, "{what}: row {k} symbol");
        assert_eq!(g.1, w.1, "{what}: row {k} position");
        assert_eq!(g.2, w.2, "{what}: row {k} bead");
    }
}

/// Centroid of atom `i` of a bead-major frame (test-side: sum over beads / P).
fn centroid(rows: &[(&str, [f64; 3])], p: usize, n: usize, i: usize) -> [f64; 3] {
    let mut c = [0.0; 3];
    for j in 0..p {
        for d in 0..3 {
            c[d] += rows[j * n + i].1[d];
        }
    }
    c.map(|v| v / p as f64)
}

fn add(x: [f64; 3], s: [f64; 3]) -> [f64; 3] {
    [x[0] + s[0], x[1] + s[1], x[2] + s[2]]
}

/// Checks a `--beads` and a centroid conversion of `frames` (P, N) against per-frame, per-atom
/// shift vectors written by hand: every bead of atom i moves by `shifts[f][i]`, and the centroid
/// output is the input centroid plus the same vector.
fn check_shifts(
    p: usize,
    n: usize,
    frames: &[InFrame],
    shifts: &[Vec<[f64; 3]>],
    args: &[&str],
    what: &str,
) {
    let d = tmp();
    let input = write(d.path(), "trajectory.xyz", &trajectory_text(p, n, frames));
    let mut a = args.to_vec();
    let cen = read_extxyz(&convert(&[&input], &d.path().join("c.extxyz"), &a));
    a.push("--beads");
    let bead = read_extxyz(&convert(&[&input], &d.path().join("b.extxyz"), &a));
    assert_eq!(cen.len(), frames.len(), "{what}: centroid frames");
    assert_eq!(bead.len(), frames.len(), "{what}: bead frames");
    for (f, (step, rows)) in frames.iter().enumerate() {
        let want_c: Vec<_> = (0..n)
            .map(|i| (rows[i].0, add(centroid(rows, p, n, i), shifts[f][i]), None))
            .collect();
        assert_eq!(cen[f].step(), *step, "{what}: step");
        assert_rows(&cen[f], &want_c, &format!("{what}: centroids, step {step}"));
        let want_b: Vec<_> = (0..p * n)
            .map(|k| (rows[k].0, add(rows[k].1, shifts[f][k % n]), Some(k / n)))
            .collect();
        assert_eq!(bead[f].step(), *step, "{what}: step");
        assert_rows(&bead[f], &want_b, &format!("{what}: beads, step {step}"));
    }
}

// ---------------------------------------------------------------------------------------------
// Required test 1: format
// ---------------------------------------------------------------------------------------------

/// M9b test 1, C-3 (wrap interval), C-2 (bonds): the hand-written 3-frame, P = 2, N = 4 input
/// (two 2-atom molecules) gives byte for byte the hand-written output for centroids, `--beads`,
/// `--every 2`, `--no-wrap` and no box (derivations in `traj_common`).
#[test]
fn format_byte_for_byte() {
    let d = tmp();
    let input = write(d.path(), "trajectory.xyz", INPUT);
    for (name, args, want) in VARIANTS {
        let out = d.path().join(format!("{name}.extxyz"));
        let got = convert(&[&input], &out, args);
        assert_eq!(
            got, want,
            "variant {name}: output differs from the hand-written text"
        );
    }
}

/// M9b test 1 (`--every`): n = 1 is the default (the same bytes as no flag), and n = 3 keeps only
/// the frame with index 0 (step 10); n larger than the frame count keeps that frame too.
#[test]
fn every_default_and_large() {
    let d = tmp();
    let input = write(d.path(), "trajectory.xyz", INPUT);
    let got = convert(
        &[&input],
        &d.path().join("a.extxyz"),
        &["--every", "1", "--box-angstrom", BOX],
    );
    assert_eq!(got, CENTROIDS, "--every 1");
    // The first 6 lines of CENTROIDS are the step-10 frame.
    let first: String = CENTROIDS
        .lines()
        .take(6)
        .map(|l| format!("{l}\n"))
        .collect();
    for n in ["3", "7"] {
        let got = convert(
            &[&input],
            &d.path().join(format!("e{n}.extxyz")),
            &["--every", n, "--box-angstrom", BOX],
        );
        assert_eq!(got, first, "--every {n}");
    }
}

// ---------------------------------------------------------------------------------------------
// Required test 2: wrapping
// ---------------------------------------------------------------------------------------------

/// M9b test 2, C-3: a 3-atom molecule (O, H, H; P = 4) straddling the x = L and y = 0 faces stays
/// whole. Step 1: O centroid (9.75, 0.25, 5) is inside, H1 (10.5, 0.75, 5) and H2 (9.5, −0.5, 5)
/// are outside, shift 0 (per-atom wrapping would move H1 and H2). Step 2: everything + (0.5, −0.5,
/// 0), O centroid (10.25, −0.25, 5) → shift (−10, +10, 0) for all beads. Step 3: everything +
/// (−10.5, 0, 12), O centroid (−0.75, 0.25, 17) → shift (+10, 0, −10).
/// Bonds (frame 1): |O−H1| = |(0.75, 0.5, 0)| = 0.90, |O−H2| = |(−0.25, −0.75, 0)| = 0.79.
#[test]
fn straddling_molecule_stays_whole() {
    let base = vec![
        // bead 0
        r("O", 9.5, 0.25, 5.0),
        r("H", 10.5, 0.75, 5.0),
        r("H", 9.5, 0.25, 5.0),
        // bead 1
        r("O", 9.75, 0.25, 5.0),
        r("H", 10.75, 0.75, 5.0),
        r("H", 9.5, -0.75, 5.0),
        // bead 2
        r("O", 10.0, 0.25, 5.0),
        r("H", 10.5, 0.75, 5.0),
        r("H", 9.5, -0.5, 5.0),
        // bead 3
        r("O", 9.75, 0.25, 5.0),
        r("H", 10.25, 0.75, 5.0),
        r("H", 9.5, -1.0, 5.0),
    ];
    let moved = |s: [f64; 3]| {
        base.iter()
            .map(|&(a, x)| (a, add(x, s)))
            .collect::<Vec<_>>()
    };
    let frames = vec![
        (100, base.clone()),
        (200, moved([0.5, -0.5, 0.0])),
        (300, moved([-10.5, 0.0, 12.0])),
    ];
    let shifts = vec![
        vec![[0.0, 0.0, 0.0]; 3],
        vec![[-10.0, 10.0, 0.0]; 3],
        vec![[10.0, 0.0, -10.0]; 3],
    ];
    check_shifts(
        4,
        3,
        &frames,
        &shifts,
        &["--box-angstrom", "10"],
        "straddling",
    );
}

/// M9b test 2, C-3: a molecule drifted by +2L (x) and −2L (y) comes back with one shift applied to
/// all its beads, so its output equals frame 1's; a lone atom elsewhere is untouched.
/// Molecule {0, 1}: centroids (5.5, 3, 4) and (6.5, 3, 4) (|d| = 1 < 1.6); atom 2 at (2, 8, 8).
/// Frame 2: molecule + (20, 0, 0) → c = (25.5, 3, 4), shift (−20, 0, 0). Frame 3: molecule +
/// (0, −20, 0) → c = (5.5, −17, 4), floor(−1.7) = −2, shift (0, +20, 0).
#[test]
fn molecule_drifted_two_boxes() {
    let mol = |s: [f64; 3]| {
        vec![
            r("C", 5.25 + s[0], 3.0 + s[1], 4.0 + s[2]),
            r("O", 6.25 + s[0], 3.0 + s[1], 4.0 + s[2]),
            r("He", 2.0, 8.0, 8.0),
            r("C", 5.75 + s[0], 3.0 + s[1], 4.0 + s[2]),
            r("O", 6.75 + s[0], 3.0 + s[1], 4.0 + s[2]),
            r("He", 2.0, 8.0, 8.25),
        ]
    };
    let frames = vec![
        (5, mol([0.0; 3])),
        (6, mol([20.0, 0.0, 0.0])),
        (7, mol([0.0, -20.0, 0.0])),
    ];
    let z = [0.0; 3];
    let shifts = vec![
        vec![z; 3],
        vec![[-20.0, 0.0, 0.0], [-20.0, 0.0, 0.0], z],
        vec![[0.0, 20.0, 0.0], [0.0, 20.0, 0.0], z],
    ];
    check_shifts(2, 3, &frames, &shifts, &["--box-angstrom", "10"], "drifted");
}

/// M9b test 2, C-3: ring polymers are never split (P = 4, three lone atoms, hand-written output).
/// Atom 0: beads x = −0.25, 0.25, 0.75, 0.25, centroid 0.25 ∈ [0, L): unchanged although bead 0 is
/// outside. Atom 1: beads x = 10.25, 9.75, 9.5, 10.5, centroid exactly 10 = L ∉ [0, L): all beads
/// − 10. Atom 2: beads x = −0.5, 0.5, 0.25, −0.25, centroid exactly 0 ∈ [0, L): unchanged.
#[test]
fn ring_polymer_not_split() {
    let rows = vec![
        r("Ne", -0.25, 2.0, 2.0),
        r("Ne", 10.25, 5.0, 5.0),
        r("Ar", -0.5, 8.0, 8.0),
        r("Ne", 0.25, 2.0, 2.0),
        r("Ne", 9.75, 5.0, 5.0),
        r("Ar", 0.5, 8.0, 8.0),
        r("Ne", 0.75, 2.0, 2.0),
        r("Ne", 9.5, 5.0, 5.0),
        r("Ar", 0.25, 8.0, 8.0),
        r("Ne", 0.25, 2.0, 2.0),
        r("Ne", 10.5, 5.0, 5.0),
        r("Ar", -0.25, 8.0, 8.0),
    ];
    let d = tmp();
    let input = write(d.path(), "t.xyz", &trajectory_text(4, 3, &[(1, rows)]));
    let b = read_extxyz(&convert(
        &[&input],
        &d.path().join("b.extxyz"),
        &["--beads", "--box-angstrom", "10"],
    ));
    let want = [
        ("Ne", [-0.25, 2.0, 2.0], Some(0)),
        ("Ne", [0.25, 5.0, 5.0], Some(0)),
        ("Ar", [-0.5, 8.0, 8.0], Some(0)),
        ("Ne", [0.25, 2.0, 2.0], Some(1)),
        ("Ne", [-0.25, 5.0, 5.0], Some(1)),
        ("Ar", [0.5, 8.0, 8.0], Some(1)),
        ("Ne", [0.75, 2.0, 2.0], Some(2)),
        ("Ne", [-0.5, 5.0, 5.0], Some(2)),
        ("Ar", [0.25, 8.0, 8.0], Some(2)),
        ("Ne", [0.25, 2.0, 2.0], Some(3)),
        ("Ne", [0.5, 5.0, 5.0], Some(3)),
        ("Ar", [-0.25, 8.0, 8.0], Some(3)),
    ];
    assert_eq!(b.len(), 1);
    assert_rows(&b[0], &want, "ring polymers");
    let c = read_extxyz(&convert(
        &[&input],
        &d.path().join("c.extxyz"),
        &["--box-angstrom", "10"],
    ));
    let want = [
        ("Ne", [0.25, 2.0, 2.0], None),
        ("Ne", [0.0, 5.0, 5.0], None),
        ("Ar", [0.0, 8.0, 8.0], None),
    ];
    assert_rows(&c[0], &want, "centroids");
}

/// M9b test 2, C-2, Q40 item 3: bonds use the minimum image, and a molecule split in the first
/// frame is made whole. P = 1, L = 10. Frame 1: atoms 0 (0.25, 5, 5) and 1 (9.5, 5, 5) are 9.25
/// apart, 0.75 under the minimum image, so one molecule; d = 9.25, k = −floor(0.925 + 0.5) = −1:
/// offset n_1 = (−10, 0, 0). Likewise 3 (2, 2, 9.75) and 4 (2, 2, 0.5): d_z = −9.25,
/// k = −floor(−0.425) = 1, n_4 = (0, 0, +10). Atom 2 (5, 5, 5) alone.
/// Frame 1 (wrap shifts 0, roots 0 and 3 inside): 1 → (−0.5, 5, 5), 4 → (2, 2, 10.5).
/// Frame 2: 0 → (−0.25, 5, 5), 1 → (9, 5, 5): wrap + (10, 0, 0) from atom 0, so 0 → 9.75 and
/// 1 → 9 − 10 + 10 = 9 (total 0); 3 → (2, 2, 10.25), 4 → (2, 2, 1): wrap − (0, 0, 10) from atom 3,
/// so 3 → 0.25 and 4 → 1 + 10 − 10 = 1 (total 0). Without the minimum image atoms 1 and 4 would be
/// their own molecules and stay at 9.5 and 0.5 in frame 1.
#[test]
fn bonds_use_minimum_image() {
    let f1 = vec![
        r("H", 0.25, 5.0, 5.0),
        r("H", 9.5, 5.0, 5.0),
        r("O", 5.0, 5.0, 5.0),
        r("H", 2.0, 2.0, 9.75),
        r("H", 2.0, 2.0, 0.5),
    ];
    let f2 = vec![
        r("H", -0.25, 5.0, 5.0),
        r("H", 9.0, 5.0, 5.0),
        r("O", 5.0, 5.0, 5.0),
        r("H", 2.0, 2.0, 10.25),
        r("H", 2.0, 2.0, 1.0),
    ];
    let z = [0.0; 3];
    let x = [10.0, 0.0, 0.0];
    let mx = [-10.0, 0.0, 0.0];
    let pz = [0.0, 0.0, 10.0];
    let mz = [0.0, 0.0, -10.0];
    check_shifts(
        1,
        5,
        &[(0, f1), (1, f2)],
        &[vec![z, mx, z, z, pz], vec![x, z, z, mz, z]],
        &["--box-angstrom", "10"],
        "minimum-image bonds",
    );
}

/// M9b test 2, Q40 item 3 (make-whole): a 3-atom chain split across the box in the first frame
/// comes out whole; its middle atom is the one across the boundary, and its beads straddle it.
/// P = 2, L = 4, bond 1.6. Frame 1 centroids: atom 0 x = 3.5 (beads 3.25, 3.75), atom 1 x = 0.5
/// (beads −0.75, 1.75: the ring polymer straddles x = 0), atom 2 x = 1.75 (beads 1.5, 2); y = z = 2.
/// Bonds (C-2): 0–1 d = −3 → 1.0; 1–2 d = 1.25; 0–2 d = −1.75 → 1.75 ≥ 1.6, no bond. Walk from 0:
/// atom 1 from 0, k = −floor(−0.75 + 0.5) = 1, n_1 = +4; atom 2 from 1, d = 1.25, k = 0,
/// n_2 = n_1 = +4 (relative to atom 0 instead: d = −1.75, k = 0, n_2 = 0, which would leave atom 2
/// at 1.75). Totals (offset + wrap) per atom:
/// - frame 1: wrap 0 (root 3.5): (0, +4, +4) → centroids 3.5, 4.5, 5.75; atom 1's beads 3.25, 5.75
///   (per-bead images of the parent would give 3.25 and 1.75, splitting it).
/// - frame 2: all + 1 in x; root 4.5 → wrap −4: totals (−4, 0, 0) → 0.5, 1.5, 2.75.
/// - frame 3: the chain stretched (atom 1 at 1.75, beads 0.5, 3; atom 2 at 3, beads 2.75, 3.25;
///   root at 3.5): offsets stay those of frame 1, totals (0, +4, +4). Offsets recomputed from this
///   frame would give atom 1 d = −1.75, k = 0, offset 0.
///
/// With `--no-wrap`: no offsets and no wrap (positions as written).
#[test]
fn split_chain_made_whole() {
    let f = |a0: [f64; 2], a1: [f64; 2], a2: [f64; 2]| {
        vec![
            r("C", a0[0], 2.0, 2.0),
            r("C", a1[0], 2.0, 2.0),
            r("C", a2[0], 2.0, 2.0),
            r("C", a0[1], 2.0, 2.0),
            r("C", a1[1], 2.0, 2.0),
            r("C", a2[1], 2.0, 2.0),
        ]
    };
    let frames = [
        (1, f([3.25, 3.75], [-0.75, 1.75], [1.5, 2.0])),
        (2, f([4.25, 4.75], [0.25, 2.75], [2.5, 3.0])),
        (3, f([3.25, 3.75], [0.5, 3.0], [2.75, 3.25])),
    ];
    let z = [0.0; 3];
    let x = [4.0, 0.0, 0.0];
    let mx = [-4.0, 0.0, 0.0];
    check_shifts(
        2,
        3,
        &frames,
        &[vec![z, x, x], vec![mx, z, z], vec![z, x, x]],
        &["--box-angstrom", "4"],
        "split chain",
    );
    check_shifts(
        2,
        3,
        &frames,
        &[vec![z; 3], vec![z; 3], vec![z; 3]],
        &["--box-angstrom", "4", "--no-wrap"],
        "split chain, --no-wrap",
    );
}

/// M9b test 2: molecules are the connected components of the FIRST kept frame, fixed afterwards;
/// a chain counts as one molecule; each molecule's first atom is its lowest index. P = 1, L = 10.
/// Frame 1: chain X = {0 (1, 2, 5), 2 (2.5, 2, 5), 4 (4, 2, 5)} (neighbours 1.5 apart, |0−4| = 3),
/// Y = {1 (1, 7, 5), 3 (2.5, 7, 5)}, X–Y ≥ 5 apart. Frame 2: X − (1.5, 0, 0) → atom 0 at x = −0.5:
/// all of X + (10, 0, 0) → 9.5, 11, 12.5; Y moved to y = 3 (1 Å from atom 2, which must not merge
/// it into X): first atom 1 at (1, 3, 5) inside, shift 0. Frame 3: X at 8.5, 10, 11.5 (first atom
/// inside, shift 0); Y at 9.25, 10.75 (first atom 1 inside, shift 0; atom 3 outside stays).
#[test]
fn molecules_from_first_frame_and_chains() {
    let f = |x0: f64, yy: f64, xy: f64| {
        vec![
            r("C", x0, 2.0, 5.0),
            r("N", xy, yy, 5.0),
            r("C", x0 + 1.5, 2.0, 5.0),
            r("N", xy + 1.5, yy, 5.0),
            r("C", x0 + 3.0, 2.0, 5.0),
        ]
    };
    let z = [0.0; 3];
    let x = [10.0, 0.0, 0.0];
    check_shifts(
        1,
        5,
        &[
            (1, f(1.0, 7.0, 1.0)),
            (2, f(-0.5, 3.0, 1.0)),
            (3, f(8.5, 7.0, 9.25)),
        ],
        &[vec![z; 5], vec![x, z, x, z, x], vec![z; 5]],
        &["--box-angstrom", "10"],
        "first-frame molecules",
    );
}

/// M9b test 2, `--bond-angstrom`: atoms 0 (0.5, 5, 5) and 1 (2.25, 5, 5) are 1.75 apart (exact).
/// Frame 2: 0 → (−0.5, 5, 5), 1 → (1.25, 5, 5). Default 1.6: two molecules, atom 0 + 10 only.
/// `--bond-angstrom 2`: one molecule, both + 10 (1 → 11.25). `--bond-angstrom 1.75`: "closer
/// than" is strict, two molecules.
#[test]
fn bond_cutoff_option() {
    let frames = [
        (0, vec![r("O", 0.5, 5.0, 5.0), r("O", 2.25, 5.0, 5.0)]),
        (1, vec![r("O", -0.5, 5.0, 5.0), r("O", 1.25, 5.0, 5.0)]),
    ];
    let z = [0.0; 3];
    let x = [10.0, 0.0, 0.0];
    let separate = [vec![z, z], vec![x, z]];
    let joined = [vec![z, z], vec![x, x]];
    check_shifts(
        1,
        2,
        &frames,
        &separate,
        &["--box-angstrom", "10"],
        "default bond",
    );
    check_shifts(
        1,
        2,
        &frames,
        &joined,
        &["--box-angstrom", "10", "--bond-angstrom", "2"],
        "bond 2",
    );
    check_shifts(
        1,
        2,
        &frames,
        &separate,
        &["--box-angstrom", "10", "--bond-angstrom", "1.75"],
        "bond 1.75 (strict)",
    );
}

// ---------------------------------------------------------------------------------------------
// Required test 3: restart joining
// ---------------------------------------------------------------------------------------------

/// P = 2, N = 1 frames whose centroid is (x, 0, 0) (beads x ∓ 0.25), one per (step, x).
fn line_frames(steps_x: &[(u64, f64)]) -> String {
    let frames: Vec<_> = steps_x
        .iter()
        .map(|&(s, x)| {
            (
                s,
                vec![r("Ar", x - 0.25, 0.0, 0.0), r("Ar", x + 0.25, 0.0, 0.0)],
            )
        })
        .collect();
    trajectory_text(2, 1, &frames)
}

/// Output (step, centroid x) pairs.
fn steps_x(text: &str) -> Vec<(u64, f64)> {
    read_extxyz(text)
        .iter()
        .map(|f| {
            assert_eq!(f.rows.len(), 1);
            (f.step(), f.rows[0].1[0])
        })
        .collect()
}

/// M9b test 3: file A holds steps 10 … 40, file B (after a restart from step 20) 30 … 50 with
/// different positions. Each step appears once: A's frames below B's first step (10, 20), then all
/// of B (A's 30 is dropped by "≥", a `>` rule would keep it twice).
#[test]
fn restart_joining_overlap() {
    let d = tmp();
    let a = write(
        d.path(),
        "a.xyz",
        &line_frames(&[(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)]),
    );
    let b = write(
        d.path(),
        "b.xyz",
        &line_frames(&[(30, 103.0), (40, 104.0), (50, 105.0)]),
    );
    let got = steps_x(&convert(&[&a, &b], &d.path().join("o.extxyz"), &[]));
    assert_eq!(
        got,
        vec![(10, 1.0), (20, 2.0), (30, 103.0), (40, 104.0), (50, 105.0)]
    );
}

/// M9b test 3: three files; the middle one ends in a partial frame (count line, header and one of
/// its two atom lines, or the count line alone), which is dropped silently; file C starts at a
/// step (55) that is not in B, so B keeps the frames below 55.
#[test]
fn restart_joining_three_files_partial_frame() {
    for partial in [
        "2\nstep=60 beads=2 atoms=1 units=angstrom\nAr 5 0 0\n",
        "2\n",
    ] {
        let d = tmp();
        let a = write(
            d.path(),
            "a.xyz",
            &line_frames(&[(10, 1.0), (20, 2.0), (30, 3.0)]),
        );
        let b_text = line_frames(&[(30, 13.0), (40, 14.0), (50, 15.0)]) + partial;
        let b = write(d.path(), "b.xyz", &b_text);
        let c = write(d.path(), "c.xyz", &line_frames(&[(55, 25.5), (65, 26.5)]));
        let got = steps_x(&convert(&[&a, &b, &c], &d.path().join("o.extxyz"), &[]));
        assert_eq!(
            got,
            vec![
                (10, 1.0),
                (20, 2.0),
                (30, 13.0),
                (40, 14.0),
                (50, 15.0),
                (55, 25.5),
                (65, 26.5)
            ],
            "partial frame {partial:?}"
        );
    }
}

/// M9b test 3 with `--every 2` (Q40): kept frames are counted from 0 over the joined sequence.
/// A: 10 … 50 (x = 1 … 5); B from step 40 (x = 104 … 106). Joined: 10, 20, 30 (A), 40, 50, 60 (B);
/// every second from index 0: 10, 30, 50 (B). Counting per file would give 10, 30, 40, 60.
#[test]
fn restart_joining_every() {
    let d = tmp();
    let a = write(
        d.path(),
        "a.xyz",
        &line_frames(&[(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0), (50, 5.0)]),
    );
    let b = write(
        d.path(),
        "b.xyz",
        &line_frames(&[(40, 104.0), (50, 105.0), (60, 106.0)]),
    );
    let got = steps_x(&convert(
        &[&a, &b],
        &d.path().join("o.extxyz"),
        &["--every", "2"],
    ));
    assert_eq!(got, vec![(10, 1.0), (30, 3.0), (50, 105.0)]);
}

/// M9b test 3, C-3: molecules found in the first file's first frame wrap the later files' frames.
/// N = 2 (P = 1), atoms 0.5 Å apart; in file B the pair sits at x = −0.25 and 0.25: both + 10.
#[test]
fn restart_joining_keeps_molecules() {
    let d = tmp();
    let a = trajectory_text(
        1,
        2,
        &[(1, vec![r("H", 5.0, 5.0, 5.0), r("H", 5.5, 5.0, 5.0)])],
    );
    let b = trajectory_text(
        1,
        2,
        &[(2, vec![r("H", -0.25, 5.0, 5.0), r("H", 0.25, 5.0, 5.0)])],
    );
    let (a, b) = (write(d.path(), "a.xyz", &a), write(d.path(), "b.xyz", &b));
    let f = read_extxyz(&convert(
        &[&a, &b],
        &d.path().join("o.extxyz"),
        &["--box-angstrom", "10"],
    ));
    assert_eq!(f.len(), 2);
    assert_rows(
        &f[1],
        &[
            ("H", [9.75, 5.0, 5.0], None),
            ("H", [10.25, 5.0, 5.0], None),
        ],
        "file B",
    );
}

/// M9b test 3, 5: P or N that differs between files is an error naming the later file and the
/// frame (its count or header line, Q40): A has P = 2, N = 1; B has P = 1, N = 2 (same P·N), or
/// P = 3, N = 1. The mismatching frame is B's first (lines 1–2), or B's second (lines 5–6) after
/// a matching first frame (P = 2, N = 1: 4 lines).
#[test]
fn restart_joining_mismatch_is_error() {
    let a_text = line_frames(&[(10, 1.0), (20, 2.0)]);
    let pn1 = trajectory_text(
        1,
        2,
        &[(30, vec![r("Ar", 1.0, 0.0, 0.0), r("Ar", 2.0, 0.0, 0.0)])],
    );
    let p3 = trajectory_text(
        3,
        1,
        &[(
            30,
            vec![
                r("Ar", 1.0, 0.0, 0.0),
                r("Ar", 2.0, 0.0, 0.0),
                r("Ar", 3.0, 0.0, 0.0),
            ],
        )],
    );
    let second = line_frames(&[(30, 3.0)]) + &p3.replace("step=30", "step=40");
    for (name, b_text, lines) in [
        ("N differs", pn1, [1, 2]),
        ("P differs", p3, [1, 2]),
        ("second frame", second, [5, 6]),
    ] {
        let d = tmp();
        let a = write(d.path(), "a.xyz", &a_text);
        let b = write(d.path(), "b.xyz", &b_text);
        let o = run_traj(&[&a, &b], &d.path().join("o.extxyz"), &[]);
        let msg = expect_clean_failure(&o, name);
        assert_names(&msg, &b, &lines, name);
    }
}

// ---------------------------------------------------------------------------------------------
// Required test 5: errors
// ---------------------------------------------------------------------------------------------

/// `INPUT` with line `k` (1-based) replaced by `new`.
fn replace_line(k: usize, new: &str) -> String {
    INPUT
        .lines()
        .enumerate()
        .map(|(i, l)| format!("{}\n", if i + 1 == k { new } else { l }))
        .collect()
}

/// The first `k` lines of `INPUT`.
fn first_lines(k: usize) -> String {
    INPUT.lines().take(k).map(|l| format!("{l}\n")).collect()
}

/// Runs the converter on `text` (as the only file, or as the second after a valid first file
/// holding steps 1 and 2 of the same P, N) and checks the error names that file and one of
/// `lines`.
fn expect_error(text: &str, lines: &[usize], what: &str) {
    let d = tmp();
    let f = write(d.path(), "bad.xyz", text);
    let o = run_traj(&[&f], &d.path().join("o.extxyz"), &["--box-angstrom", BOX]);
    let msg = expect_clean_failure(&o, what);
    assert_names(&msg, &f, lines, what);
    // Second file: line numbers count within that file. Its frames come after steps 1 and 2.
    let first = INPUT
        .replace("step=10 ", "step=1 ")
        .replace("step=20 ", "step=2 ")
        .lines()
        .take(20)
        .map(|l| format!("{l}\n"))
        .collect::<String>();
    let a = write(d.path(), "first.xyz", &first);
    let o = run_traj(&[&a, &f], &d.path().join("o2.extxyz"), &[]);
    let msg = expect_clean_failure(&o, what);
    assert_names(&msg, &f, lines, &format!("{what} (second file)"));
}

/// M9b test 5: a malformed header names the file and the frame's header line (or its count line,
/// Q40): missing `units=`, a non-numeric step, a missing `step=`.
#[test]
fn error_malformed_header() {
    for (k, bad) in [
        (12, "step=20 beads=2 atoms=4"),
        (12, "step=2x beads=2 atoms=4 units=angstrom"),
        (2, "beads=2 atoms=4 units=angstrom"),
        (22, "step=30 beads=two atoms=4 units=angstrom"),
    ] {
        expect_error(
            &replace_line(k, bad),
            &[k - 1, k],
            &format!("header {bad:?}"),
        );
    }
}

/// M9b test 5: a malformed atom line names exactly that line: too few fields, a non-numeric
/// coordinate, an empty line in place of an atom line.
#[test]
fn error_malformed_atom_line() {
    for (k, bad) in [
        (15, "H 6.25 -3.25"),
        (25, "F 7 x 28.5"),
        (3, "1 2 3"),
        (14, ""),
    ] {
        expect_error(&replace_line(k, bad), &[k], &format!("atom line {bad:?}"));
    }
}

/// M9b test 5: a count line that is not a number, or ≠ P·N (7 and 9 for P·N = 8), names the
/// frame's count line (or header line, Q40).
#[test]
fn error_count_line() {
    for (k, bad) in [(21, "eight"), (11, "7"), (1, "9"), (21, "")] {
        expect_error(
            &replace_line(k, bad),
            &[k, k + 1],
            &format!("count {bad:?}"),
        );
    }
}

/// M9b test 5: a step that does not increase within a file (equal, or smaller) names the frame
/// (its count or header line, Q40).
#[test]
fn error_step_not_increasing() {
    for bad in [
        "step=20 beads=2 atoms=4 units=angstrom",
        "step=15 beads=2 atoms=4 units=angstrom",
    ] {
        expect_error(&replace_line(22, bad), &[21, 22], bad);
    }
}

/// M9b test 5: a partial last frame in the last file is an error naming the file and the frame
/// (count or header line) or where it ends (the last line, or the line after it; Q40). Cuts:
/// after 27 lines (5 of 8 atom lines), after 22 (header only), after 21 (count line only).
#[test]
fn error_partial_last_frame() {
    for k in [27, 22, 21] {
        expect_error(
            &first_lines(k),
            &[21, 22, k, k + 1],
            &format!("cut after {k} lines"),
        );
    }
}

/// M9b test 5: the run stops on the first error; with bad lines 15 and 25 the message names 15
/// and not 25.
#[test]
fn error_first_error_wins() {
    let text = replace_line(15, "H 6.25 oops 8");
    let text: String = text
        .lines()
        .enumerate()
        .map(|(i, l)| format!("{}\n", if i + 1 == 25 { "F 7" } else { l }))
        .collect();
    let d = tmp();
    let f = write(d.path(), "bad.xyz", &text);
    let o = run_traj(&[&f], &d.path().join("o.extxyz"), &[]);
    let msg = expect_clean_failure(&o, "two errors");
    assert_names(&msg, &f, &[15], "two errors");
    assert!(
        !mentions_line(&msg, &f, 25),
        "names the second error: {msg}"
    );
}

/// M9b test 5: L ≤ 0 or non-finite is an error (non-zero exit, not a panic).
#[test]
fn error_bad_box() {
    let d = tmp();
    let input = write(d.path(), "trajectory.xyz", INPUT);
    for l in ["0", "-10", "inf", "NaN", "-0"] {
        let o = run_traj(
            &[&input],
            &d.path().join("o.extxyz"),
            &["--box-angstrom", l],
        );
        expect_clean_failure(&o, &format!("--box-angstrom {l}"));
    }
}

/// M9b test 5: `--every 0` is an error (non-zero exit, not a panic), with or without a box.
#[test]
fn error_every_zero() {
    let d = tmp();
    let input = write(d.path(), "trajectory.xyz", INPUT);
    for args in [
        &["--every", "0"][..],
        &["--every", "0", "--box-angstrom", BOX][..],
    ] {
        let o = run_traj(&[&input], &d.path().join("o.extxyz"), args);
        expect_clean_failure(&o, &format!("{args:?}"));
    }
}

// ---------------------------------------------------------------------------------------------
// Required test 6: end-to-end
// ---------------------------------------------------------------------------------------------

/// Four neon atoms in a 12.5 Å box (P = 4, LJ as M4's BASE), placed so that centroids lie outside
/// [0, L) from the start: atom 2 at y = −1.5 (+L in y), atom 3 at z = 13 (−L in z). Atom 0 at
/// x = 11.5 and atom 1 at x = 2.1 are 3.1 Å apart only under the minimum image, so with
/// `--bond-angstrom 3.5` they form a molecule split across the box (atom 1 gets offset +L, Q40
/// item 3). Minimum-image distances between atoms are ≥ 3.1 Å (0–1), the rest > 5 Å.
const E2E: &str = r#"
[system]
n_atoms = 4
mass_amu = 20.1797
box_angstrom = 12.5
positions_angstrom = [[11.5, 6.0, 6.0], [2.1, 6.0, 6.0], [6.0, -1.5, 6.0], [9.0, 9.0, 13.0]]

[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.0
skin_sigma = 0.25
shift = true

[sampling]
temperature_kelvin = 30.0
n_beads = 4
dt_fs = 1.0
tau0_fs = 100.0
n_equil = 10
n_steps = 60
stride = 5
seed = 2026

[output]
checkpoint_every = 30
trajectory_every = 5
"#;

/// Test-side reader of `trajectory.xyz`: (step, P·N positions in Å, bead-major).
fn read_input(path: &Path, p: usize, n: usize) -> Vec<(u64, Vec<[f64; 3]>)> {
    let t = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = t.lines().collect();
    assert_eq!(lines.len() % (p * n + 2), 0);
    lines
        .chunks(p * n + 2)
        .map(|f| {
            let step = f[1]
                .split(' ')
                .next()
                .unwrap()
                .strip_prefix("step=")
                .unwrap()
                .parse()
                .unwrap();
            let x = f[2..]
                .iter()
                .map(|l| {
                    let w: Vec<f64> = l.split(' ').skip(1).map(|v| v.parse().unwrap()).collect();
                    [w[0], w[1], w[2]]
                })
                .collect();
            (step, x)
        })
        .collect()
}

/// Independent checker (C-2, C-3, Q40 item 3 written here): molecules and make-whole offsets
/// from the first frame's centroids by a FIFO breadth-first walk from each lowest-index unvisited
/// atom, visiting bonded atoms (minimum-image distance < `bond`) in ascending index; a reached atom
/// j gets its parent's offset plus −floor(d/L + 1/2)·L, d = c_j − c_parent. Wrap shift per
/// molecule = −floor(c/L)·L of its root's centroid (root offset 0). Returns per frame, per atom
/// (centroid, total shift = offset + wrap, offset).
#[allow(clippy::type_complexity)]
fn checker(
    frames: &[(u64, Vec<[f64; 3]>)],
    p: usize,
    n: usize,
    l: f64,
    bond: f64,
) -> Vec<Vec<([f64; 3], [f64; 3], [f64; 3])>> {
    let cent = |x: &Vec<[f64; 3]>| -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let mut c = [0.0; 3];
                for j in 0..p {
                    for d in 0..3 {
                        c[d] += x[j * n + i][d];
                    }
                }
                c.map(|v| v / p as f64)
            })
            .collect()
    };
    let c0 = cent(&frames[0].1);
    let mi = |d: f64| d - l * (d / l + 0.5).floor();
    let mut root = vec![usize::MAX; n];
    let mut off = vec![[0.0f64; 3]; n];
    for s in 0..n {
        if root[s] != usize::MAX {
            continue;
        }
        root[s] = s;
        let mut queue = std::collections::VecDeque::from([s]);
        while let Some(i) = queue.pop_front() {
            for j in 0..n {
                if root[j] == usize::MAX {
                    let r2: f64 = (0..3).map(|d| mi(c0[j][d] - c0[i][d]).powi(2)).sum();
                    if r2.sqrt() < bond {
                        root[j] = s;
                        for d in 0..3 {
                            let k = -((c0[j][d] - c0[i][d]) / l + 0.5).floor();
                            off[j][d] = off[i][d] + k * l;
                        }
                        queue.push_back(j);
                    }
                }
            }
        }
    }
    frames
        .iter()
        .map(|(_, x)| {
            let c = cent(x);
            (0..n)
                .map(|i| {
                    let w = c[root[i]].map(|v| -(v / l).floor() * l);
                    let o = off[i];
                    (c[i], [o[0] + w[0], o[1] + w[1], o[2] + w[2]], o)
                })
                .collect()
        })
        .collect()
}

/// M9b test 6, C-2, C-3: a short engine run (LJ, P = 4, periodic) exported with `--beads` and by
/// default (centroids), with the default bond (every atom alone) and `--bond-angstrom 3.5` (atoms 0
/// and 1, 3.1 Å apart across the box, one molecule made whole): the centroids of the bead output equal the centroid output
/// to 1e-12 Å; both match the test's own checker (input + offset + one shift per molecule) to
/// 1e-12 Å.
#[test]
fn end_to_end_engine_run() {
    let (p, n, l) = (4, 4, 12.5);
    let d = tmp();
    let cfg = pimd_cli::Config::from_toml_str(E2E).unwrap();
    pimd_cli::run(&cfg, d.path()).unwrap();
    let traj = d.path().join("trajectory.xyz");
    let input = read_input(&traj, p, n);
    assert_eq!(input.len(), 12, "frames at steps 5 … 60");
    for (bond, args) in [
        (1.6, vec!["--box-angstrom", "12.5"]),
        (
            3.5,
            vec!["--box-angstrom", "12.5", "--bond-angstrom", "3.5"],
        ),
    ] {
        let want = checker(&input, p, n, l, bond);
        let mut shifted = false;
        let mut whole = false;
        let cen = read_extxyz(&convert(&[&traj], &d.path().join("c.extxyz"), &args));
        let mut ab = args.clone();
        ab.push("--beads");
        let bead = read_extxyz(&convert(&[&traj], &d.path().join("b.extxyz"), &ab));
        assert_eq!(cen.len(), input.len());
        assert_eq!(bead.len(), input.len());
        for f in 0..input.len() {
            assert_eq!(cen[f].step(), input[f].0);
            assert_eq!(bead[f].step(), input[f].0);
            assert_eq!(cen[f].rows.len(), n);
            assert_eq!(bead[f].rows.len(), p * n);
            for i in 0..n {
                let (c, s, o) = want[f][i];
                shifted |= s != o;
                whole |= o != [0.0; 3];
                let mut from_beads = [0.0; 3];
                for j in 0..p {
                    let row = &bead[f].rows[j * n + i];
                    assert_eq!(row.0, "X");
                    assert_eq!(row.2, Some(j));
                    for k in 0..3 {
                        from_beads[k] += row.1[k] / p as f64;
                        let w = input[f].1[j * n + i][k] + s[k];
                        assert!(
                            (row.1[k] - w).abs() <= 1e-12,
                            "bond {bond}, frame {f}, bead {j}, atom {i}, dim {k}: {} vs {w}",
                            row.1[k]
                        );
                    }
                }
                for k in 0..3 {
                    let got = cen[f].rows[i].1[k];
                    assert!(
                        (got - from_beads[k]).abs() <= 1e-12,
                        "bond {bond}, frame {f}, atom {i}, dim {k}: centroid {got} vs beads {}",
                        from_beads[k]
                    );
                    assert!(
                        (got - (c[k] + s[k])).abs() <= 1e-12,
                        "bond {bond}, frame {f}, atom {i}, dim {k}: {got} vs checker {}",
                        c[k] + s[k]
                    );
                }
            }
        }
        assert!(shifted, "test setup: some centroid must need a shift");
        assert_eq!(
            whole,
            bond > 3.0,
            "test setup: make-whole offsets only with bond 3.5"
        );
    }
}
