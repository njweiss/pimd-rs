//! Tests for `LennardJones` (SPEC P-1, P-3, P-4, P-5, N-2, N-3, C-4, C-5, S-3, S-4).
//!
//! Expected values are hand-computed (fractions in the comments) or come from the test's own
//! brute-force O(N²) LJ in `common` (all pairs i < j, own minimum image), never from the code
//! under test.

mod common;

use common::{
    SplitMix64, beads, beads1, brute_lj, disp, lj_pair, min_gap_to_cutoff, norm2, random_config,
    unwrap_randomly,
};
use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, LennardJones, LjParams, Potential, PotentialError};

/// ε = 1/2 Eh, σ = 2 bohr, r_c = 2.5σ = 5 bohr, skin 1/2 bohr.
const EPS: f64 = 0.5;
const SIG: f64 = 2.0;
const RC: f64 = 5.0;
const SKIN: f64 = 0.5;

fn params(eps: f64, sig: f64, rc: f64, skin: f64, shift: bool) -> LjParams {
    LjParams {
        epsilon: eps,
        sigma: sig,
        cutoff: rc,
        skin,
        shift,
    }
}

fn lj(shift: bool) -> LennardJones {
    LennardJones::new(params(EPS, SIG, RC, SKIN, shift)).unwrap()
}

fn eval(pot: &mut LennardJones, x: &[[f64; 3]], cell: Option<&Cell>) -> Evaluation {
    pot.evaluate(&beads1(x), cell, None).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

/// V(r_c) for ε = 1/2, σ = 2, r_c = 5: (σ/r_c)⁶ = 0.4⁶ = 0.004096, (σ/r_c)¹² = 0.000016777216,
/// V(r_c) = 4·½·(0.000016777216 − 0.004096) = −0.008158445568 Eh.
const V_RC: f64 = -0.008158445568;

/// P-3, P-5, S-3, S-4: two atoms in open boundaries, separated along e = (1, 2, 2)/3 by r, with
/// hand-computed V(r) and radial force F(r) = −dV/dr = (24ε/r)[2(σ/r)¹² − (σ/r)⁶] for
/// ε = 1/2, σ = 2:
///   r = σ = 2:         V = 0;                F = 24ε/σ = 6
///   r = 2^(1/6) σ:      V = −ε = −1/2;        F = 0
///   r = 1.25σ = 2.5:   (σ/r)⁶ = 0.262144, (σ/r)¹² = 0.068719476736;
///                      V = 2(0.068719476736 − 0.262144) = −0.386849046528;
///                      F = 4.8(0.137438953472 − 0.262144) = −0.5985842233344
///   r = 1.5σ = 3:      (σ/r)⁶ = 64/729, (σ/r)¹² = 4096/531441;
///                      V = 2(4096 − 46656)/531441 = −85120/531441;
///                      F = 4(8192 − 46656)/531441 = −153856/531441
///   r = 2σ = 4:        V = 4ε(1/4096 − 1/64) = −63/2048;  F = 3(2/4096 − 1/64) = −93/2048
/// The force on atom 0 (at the origin) is F(r)·(r_0 − r_1)/r = −F e, on atom 1 +F e (Newton's third
/// law). Virial (E-8, P-5): r_01 = −r e, f_01 = −F e, so W = r F e ⊗ e. Unshifted energies are V(r);
/// shifted ones V(r) − V(r_c) with V(r_c) above; forces and virial are the same in both (P-3:
/// forces never shifted). Tolerance 1e-13 relative (floor 1): a few roundings of O(1) numbers.
#[test]
fn two_atoms_hand_computed() {
    let r6 = 2f64.powf(1.0 / 6.0) * SIG;
    let cases = [
        (2.0, 0.0, 6.0),
        (r6, -0.5, 0.0),
        (2.5, -0.386849046528, -0.5985842233344),
        (3.0, -85120.0 / 531441.0, -153856.0 / 531441.0),
        (4.0, -63.0 / 2048.0, -93.0 / 2048.0),
    ];
    let e = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    for shift in [false, true] {
        for &(r, v, f) in &cases {
            let x = [[0.0; 3], e.map(|c| r * c)];
            let mut pot = lj(shift);
            let ev = eval(&mut pot, &x, None);
            let want_e = if shift { v - V_RC } else { v };
            assert_eq!(ev.energies.len(), 1);
            assert!(
                close(ev.energies[0], want_e, 1e-13),
                "r = {r}, shift {shift}: E = {}, want {want_e}",
                ev.energies[0]
            );
            let (f0, f1) = (ev.forces.get(0, 0), ev.forces.get(0, 1));
            for a in 0..3 {
                assert!(
                    close(f0[a], -f * e[a], 1e-13) && close(f1[a], f * e[a], 1e-13),
                    "r = {r}, shift {shift}: forces {f0:?} {f1:?}, want ∓{f}·e"
                );
            }
            let w = ev.virial.expect("P = 1 gives a virial (P-5)");
            for a in 0..3 {
                for b in 0..3 {
                    assert!(
                        close(w[a][b], r * f * e[a] * e[b], 1e-13),
                        "r = {r}: W = {w:?}, want r F e⊗e with r F = {}",
                        r * f
                    );
                }
            }
        }
    }
}

/// P-3: the cutoff edge, two atoms along x in open boundaries (ε = 1/2, σ = 2, r_c = 5).
/// - r = r_c(1 − 1e-10), just inside: unshifted V = V(r) ≈ V(r_c) = −0.008158445568; shifted
///   V(r) − V(r_c) ≈ 0 (|·| < 1e-11); the force is the unshifted F(r) ≈ F(r_c) =
///   2.4(2·0.000016777216 − 0.004096) = −0.0097498693632 in both cases (forces never shifted), so
///   the force on atom 0 is +|F| along x (attraction towards atom 1).
/// - r = r_c exactly (atom 1 at (5, 0, 0), r² = 25 exactly): r ≥ r_c contributes nothing.
/// - r = r_c(1 + 1e-10) and r = r_c + skin/2 (still in the neighbor list): nothing, for both shift
///   settings; in particular the shift is not applied beyond r_c and the force is truncated
///   together with the energy.
///
/// Expected values for just inside are the test's own `lj_pair` at that r, to 1e-12 relative (the
/// shifted energy, a difference of nearly equal numbers, to 1e-15 absolute). "Nothing" is exactly
/// zero energy, force and virial.
#[test]
fn cutoff_edge() {
    let (v_rc, f_rc) = (V_RC, -0.0097498693632);
    for shift in [false, true] {
        let r = RC * (1.0 - 1e-10);
        let mut pot = lj(shift);
        let ev = eval(&mut pot, &[[0.0; 3], [r, 0.0, 0.0]], None);
        let (v, f_over_r) = lj_pair(EPS, SIG, r * r);
        let f = f_over_r * r;
        assert!(close(f, f_rc, 1e-8) && close(v, v_rc, 1e-8));
        if shift {
            assert!(
                (ev.energies[0] - (v - V_RC)).abs() <= 1e-15,
                "{:?}",
                ev.energies
            );
            assert!(ev.energies[0].abs() < 1e-11);
        } else {
            assert!(close(ev.energies[0], v, 1e-12), "{:?}", ev.energies);
        }
        let f0 = ev.forces.get(0, 0);
        assert!(
            close(f0[0], -f, 1e-12) && f0[0] > 0.0 && f0[1] == 0.0 && f0[2] == 0.0,
            "shift {shift}: force just inside r_c {f0:?}, want ({}, 0, 0)",
            -f
        );
        for r in [RC, RC * (1.0 + 1e-10), RC + 0.5 * SKIN] {
            let mut pot = lj(shift);
            let ev = eval(&mut pot, &[[0.0; 3], [r, 0.0, 0.0]], None);
            assert_eq!(ev.energies, vec![0.0], "r = {r}, shift {shift}");
            assert!(
                ev.forces.as_slice().iter().all(|&f| f == 0.0),
                "r = {r}, shift {shift}: {:?}",
                ev.forces
            );
            assert_eq!(ev.virial, Some([[0.0; 3]; 3]), "r = {r}, shift {shift}");
        }
    }
}

/// P-3 (no tail correction), C-1: in a periodic cell (L = 12 ≥ 2(r_c + skin) = 11) a lone atom and
/// a pair beyond r_c have energy exactly 0 (a tail correction would add a density-dependent
/// constant), and a pair inside r_c has exactly the bare pair energy V(3) = −85120/531441 (or
/// V(3) − V(r_c) shifted), the same as in open boundaries.
#[test]
fn no_tail_correction() {
    let cell = Cell::new([12.0; 3]).unwrap();
    for shift in [false, true] {
        let mut pot = lj(shift);
        assert_eq!(
            eval(&mut pot, &[[1.0, 2.0, 3.0]], Some(&cell)).energies,
            [0.0]
        );
        let mut pot = lj(shift);
        let far = eval(&mut pot, &[[1.0, 1.0, 1.0], [7.0, 1.0, 1.0]], Some(&cell));
        assert_eq!(far.energies, [0.0], "shift {shift}");
        let mut pot = lj(shift);
        let near = eval(&mut pot, &[[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]], Some(&cell));
        let v = -85120.0 / 531441.0 - if shift { V_RC } else { 0.0 };
        assert!(close(near.energies[0], v, 1e-13), "{:?}", near.energies);
    }
}

/// Parameters of the random periodic tests: ε = 0.7 Eh, σ = 1.1 bohr, r_c = 2.75 = 2.5σ,
/// skin 0.6, cell (7.3, 8.1, 6.9) so r_c + skin = 3.35 ≤ L_min/2 = 3.45 (C-4).
const RL: [f64; 3] = [7.3, 8.1, 6.9];
const R_EPS: f64 = 0.7;
const R_SIG: f64 = 1.1;
const R_RC: f64 = 2.75;
const R_SKIN: f64 = 0.6;

fn random_lj(shift: bool) -> LennardJones {
    LennardJones::new(params(R_EPS, R_SIG, R_RC, R_SKIN, shift)).unwrap()
}

/// P-3, P-5, C-2, C-5, S-3: random periodic configurations (N = 100, seeded, minimum separation
/// 0.85σ, given in unwrapped coordinates: each atom shifted by random multiples of L) against the
/// test's brute-force minimum-image LJ over all pairs i < j. Energy, every force component and
/// every virial entry agree to 1e-12 relative (floor: 1e-12 of the largest |F|, |W| entry, or of
/// max(|E|, 1)). Newton's third law: |Σ_i F_i| ≤ 1e-12 Σ_i |F_i| per component. Shift: forces equal
/// unshifted ones to 1e-14 relative and E_unshifted − E_shifted = n_inside·V(r_c) (n_inside pairs
/// with r < r_c, counted by the brute force) to 1e-12. Also open boundaries (cell `None`, points
/// spread over the same box).
#[test]
fn random_configurations_match_brute_force() {
    let cell = Cell::new(RL).unwrap();
    for (seed, periodic) in [(1u64, true), (2, true), (3, false)] {
        let mut rng = SplitMix64(seed);
        let x0 = random_config(&mut rng, 100, RL, 0.85 * R_SIG, periodic);
        let x = if periodic {
            unwrap_randomly(&mut rng, &x0, RL)
        } else {
            x0
        };
        let (c, l) = if periodic {
            (Some(&cell), Some(RL))
        } else {
            (None, None)
        };
        let mut evs = Vec::new();
        for shift in [false, true] {
            let want = brute_lj(&x, l, R_EPS, R_SIG, R_RC, shift);
            let mut pot = random_lj(shift);
            let ev = eval(&mut pot, &x, c);
            assert!(
                close(ev.energies[0], want.energy, 1e-12),
                "seed {seed} shift {shift}: E = {}, brute force {}",
                ev.energies[0],
                want.energy
            );
            let fmax = want
                .forces
                .iter()
                .flatten()
                .fold(0.0f64, |m, f| m.max(f.abs()));
            let mut sum = [0.0; 3];
            let mut abs = [0.0; 3];
            for (i, fw) in want.forces.iter().enumerate() {
                let f = ev.forces.get(0, i);
                for a in 0..3 {
                    assert!(
                        (f[a] - fw[a]).abs() <= 1e-12 * fmax,
                        "seed {seed} shift {shift} atom {i}: {f:?} vs brute force {fw:?}"
                    );
                    sum[a] += f[a];
                    abs[a] += f[a].abs();
                }
            }
            for a in 0..3 {
                assert!(sum[a].abs() <= 1e-12 * abs[a], "Σ F = {sum:?}");
            }
            let w = ev.virial.expect("P = 1 virial");
            let wmax = want
                .virial
                .iter()
                .flatten()
                .fold(0.0f64, |m, v| m.max(v.abs()));
            for a in 0..3 {
                for b in 0..3 {
                    assert!(
                        (w[a][b] - want.virial[a][b]).abs() <= 1e-12 * wmax,
                        "seed {seed}: W = {w:?}, brute force {:?}",
                        want.virial
                    );
                }
            }
            evs.push((ev, want.n_inside));
        }
        let (unshifted, shifted) = (&evs[0].0, &evs[1].0);
        let vc = lj_pair(R_EPS, R_SIG, R_RC * R_RC).0;
        let de = unshifted.energies[0] - shifted.energies[0];
        assert!(
            close(de, evs[0].1 as f64 * vc, 1e-12),
            "seed {seed}: E_unshifted − E_shifted = {de}, want {} · V(r_c)",
            evs[0].1
        );
        for (a, b) in unshifted
            .forces
            .as_slice()
            .iter()
            .zip(shifted.forces.as_slice())
        {
            assert!((a - b).abs() <= 1e-14 * b.abs().max(1.0));
        }
    }
}

/// P-3, S-3: forces equal −∂E/∂x by central finite differences (h = 1e-5 bohr) to 1e-6 relative
/// (floor ε/σ), every component of every atom, on two seeded random periodic configurations
/// (N = 40, unwrapped coordinates) and for both shift settings. The configurations are drawn
/// until no pair lies within 1e-3 bohr of r_c, so no displaced pair crosses the cutoff (where the
/// unshifted energy jumps). Error budget: O(h² V''') ≈ 1e-8 and rounding ε_mach |E| / h ≈ 1e-10.
#[test]
fn forces_match_finite_differences() {
    let cell = Cell::new(RL).unwrap();
    let h = 1e-5;
    for seed in [11u64, 12] {
        let mut rng = SplitMix64(seed);
        // Draw configurations from the seeded stream until one has no pair within 1e-3 of r_c.
        let x0 = loop {
            let x = random_config(&mut rng, 40, RL, 0.9 * R_SIG, true);
            if min_gap_to_cutoff(&x, Some(RL), R_RC) > 1e-3 {
                break x;
            }
        };
        let x = unwrap_randomly(&mut rng, &x0, RL);
        for shift in [false, true] {
            let mut pot = random_lj(shift);
            let f = eval(&mut pot, &x, Some(&cell)).forces;
            for i in 0..x.len() {
                for a in 0..3 {
                    let mut xp = x.clone();
                    xp[i][a] += h;
                    let mut xm = x.clone();
                    xm[i][a] -= h;
                    let ep = eval(&mut pot, &xp, Some(&cell)).energies[0];
                    let em = eval(&mut pot, &xm, Some(&cell)).energies[0];
                    let fd = -(ep - em) / (2.0 * h);
                    let fa = f.get(0, i)[a];
                    assert!(
                        (fa - fd).abs() <= 1e-6 * fa.abs().max(R_EPS / R_SIG),
                        "seed {seed} shift {shift} atom {i} axis {a}: F = {fa}, −dE/dx = {fd}"
                    );
                }
            }
        }
    }
}

/// N-3, P-5, S-4: one neighbor list per bead. P = 3 beads with different configurations (bead 1
/// has a pair at 1.2σ that is far apart, beyond r_c + skin, in beads 0 and 2; bead 2 is bead 0
/// shifted by random multiples of L) give per-bead energies and forces equal to separate P = 1
/// evaluations of each bead with fresh `LennardJones` objects (1e-13 relative, floor 1), and
/// `virial` is `None` for P > 1 (P-5) but `Some` for each P = 1 evaluation.
#[test]
fn one_list_per_bead() {
    let cell = Cell::new(RL).unwrap();
    let mut rng = SplitMix64(21);
    let b0 = random_config(&mut rng, 30, RL, 0.9 * R_SIG, true);
    let mut b1 = random_config(&mut rng, 30, RL, 0.9 * R_SIG, true);
    // Atoms 0 and 1: far apart in bead 0 (≥ 3.4 > r_c + skin by construction below) and close in
    // bead 1.
    let mut b0 = b0;
    b0[0] = [0.5, 0.5, 0.5];
    b0[1] = [0.5 + 3.4, 0.5, 0.5 + 3.4];
    b1[0] = [2.0, 2.0, 2.0];
    b1[1] = [2.0 + 1.2 * R_SIG, 2.0, 2.0];
    let b2 = unwrap_randomly(&mut rng, &b0, RL);
    let q = beads(&[b0.clone(), b1.clone(), b2.clone()]);
    for shift in [false, true] {
        let mut pot = random_lj(shift);
        let ev = pot.evaluate(&q, Some(&cell), None).unwrap();
        assert_eq!(ev.energies.len(), 3);
        assert_eq!((ev.forces.n_beads(), ev.forces.n_atoms()), (3, 30));
        assert_eq!(ev.virial, None, "P = 3 must give virial None (P-5)");
        for (j, conf) in [&b0, &b1, &b2].into_iter().enumerate() {
            let mut fresh = random_lj(shift);
            let one = eval(&mut fresh, conf, Some(&cell));
            assert!(one.virial.is_some());
            assert!(
                close(ev.energies[j], one.energies[0], 1e-13),
                "bead {j}: {} vs single-bead {}",
                ev.energies[j],
                one.energies[0]
            );
            for i in 0..30 {
                let (a, b) = (ev.forces.get(j, i), one.forces.get(0, i));
                for d in 0..3 {
                    assert!(
                        close(a[d], b[d], 1e-13),
                        "bead {j} atom {i}: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }
}

/// N-2, N-3, C-5: the lists are reused and rebuilt correctly across calls. One `LennardJones`
/// object evaluates a 40-step random walk (each step moves every atom by up to 0.3·skin per
/// component, so the cumulative displacement soon exceeds skin/2 and rebuilds are needed; P = 2
/// with independent walks per bead, unwrapped coordinates). Every call agrees with a fresh object
/// on the same positions (energies and forces to 1e-12 relative, floor 1).
#[test]
fn lists_reused_and_rebuilt_across_calls() {
    let cell = Cell::new(RL).unwrap();
    let mut rng = SplitMix64(31);
    let mut conf = vec![
        random_config(&mut rng, 50, RL, 0.95 * R_SIG, true),
        random_config(&mut rng, 50, RL, 0.95 * R_SIG, true),
    ];
    let mut pot = random_lj(true);
    for step in 0..40 {
        let q = beads(&conf);
        let ev = pot.evaluate(&q, Some(&cell), None).unwrap();
        let mut fresh = random_lj(true);
        let want = fresh.evaluate(&q, Some(&cell), None).unwrap();
        for j in 0..2 {
            assert!(
                close(ev.energies[j], want.energies[j], 1e-12),
                "step {step} bead {j}: reused lists give {}, fresh {}",
                ev.energies[j],
                want.energies[j]
            );
        }
        for (a, b) in ev.forces.as_slice().iter().zip(want.forces.as_slice()) {
            assert!(close(*a, *b, 1e-12), "step {step}: forces differ");
        }
        for c in conf.iter_mut() {
            for r in c.iter_mut() {
                for v in r.iter_mut() {
                    *v += rng.range(-0.3, 0.3) * R_SKIN;
                }
            }
        }
    }
}

/// N-2: a stale list is detected. Atoms 0 and 1 start 3.4 bohr apart, beyond r_c + skin = 3.35
/// (so not in each other's list); then atom 1 moves by 2.08 bohr (> skin/2) to 1.2σ from atom 0
/// while the others stay. The second evaluation must include the pair: its energy equals the
/// brute force (1e-12 relative), and differs from the first by more than 0.1 Eh (the new pair
/// alone contributes V(1.2σ) ≈ −0.62 Eh). Open boundaries (atoms 0, 1 far from the rest) and
/// periodic (no other atom within 5 bohr of atom 0), both shift settings.
#[test]
fn stale_list_is_rebuilt() {
    let cell = Cell::new(RL).unwrap();
    for periodic in [false, true] {
        let mut rng = SplitMix64(41);
        let mut x = random_config(&mut rng, 20, RL, 1.0 * R_SIG, periodic);
        let (c, l) = if periodic {
            (Some(&cell), Some(RL))
        } else {
            (None, None)
        };
        if !periodic {
            // Atoms 0 and 1 far from the rest, 3.4 apart.
            x[0] = [-30.0, -30.0, -30.0];
            x[1] = [-30.0 + 3.4, -30.0, -30.0];
        } else {
            // Atom 0 at (1, 1, 1) with no other atom within 5 bohr, atom 1 3.4 away along x.
            x.retain(|r| norm2(disp(*r, [1.0, 1.0, 1.0], l)) > 25.0);
            x.insert(0, [1.0 + 3.4, 1.0, 1.0]);
            x.insert(0, [1.0, 1.0, 1.0]);
        }
        for shift in [false, true] {
            let mut pot = random_lj(shift);
            let first = eval(&mut pot, &x, c).energies[0];
            let mut y = x.clone();
            y[1] = [y[0][0] + 1.2 * R_SIG, y[0][1], y[0][2]];
            let second = eval(&mut pot, &y, c).energies[0];
            let want = brute_lj(&y, l, R_EPS, R_SIG, R_RC, shift).energy;
            assert!(
                close(second, want, 1e-12),
                "periodic {periodic} shift {shift}: after the move E = {second}, want {want} \
                 (before: {first})"
            );
            assert!(
                (second - first).abs() > 0.1,
                "pair missing: {first} → {second}"
            );
        }
    }
}

/// N-3 (rebuild when the atom count changes), P-1: `LennardJones` accepts any N; the same object
/// evaluating N = 2, then N = 3, then N = 2 again gives the brute-force energies (1e-13).
#[test]
fn atom_count_change_rebuilds() {
    let mut pot = lj(false);
    let a = vec![[0.0; 3], [3.0, 0.0, 0.0]];
    let b = vec![[0.0; 3], [3.0, 0.0, 0.0], [0.0, 2.5, 0.0]];
    for x in [&a, &b, &a] {
        let want = brute_lj(x, None, EPS, SIG, RC, false).energy;
        let got = eval(&mut pot, x, None).energies[0];
        assert!(close(got, want, 1e-13), "N = {}: {got} vs {want}", x.len());
    }
}

/// C-5, C-2: positions are unwrapped. Shifting atoms by random multiples of L leaves energies,
/// forces and virial unchanged (1e-12 relative, floor 1) for P = 1 and P = 2.
#[test]
fn unwrapped_positions_equal_wrapped() {
    let cell = Cell::new(RL).unwrap();
    let mut rng = SplitMix64(51);
    let x0 = random_config(&mut rng, 60, RL, 0.9 * R_SIG, true);
    let x1 = random_config(&mut rng, 60, RL, 0.9 * R_SIG, true);
    let u0 = unwrap_randomly(&mut rng, &x0, RL);
    let u1 = unwrap_randomly(&mut rng, &x1, RL);
    for (w, u) in [
        (beads1(&x0), beads1(&u0)),
        (beads(&[x0.clone(), x1.clone()]), beads(&[u0, u1])),
    ] {
        let ew = random_lj(false).evaluate(&w, Some(&cell), None).unwrap();
        let eu = random_lj(false).evaluate(&u, Some(&cell), None).unwrap();
        for (a, b) in ew.energies.iter().zip(&eu.energies) {
            assert!(close(*a, *b, 1e-12), "{a} vs {b}");
        }
        for (a, b) in ew.forces.as_slice().iter().zip(eu.forces.as_slice()) {
            assert!(close(*a, *b, 1e-12));
        }
        match (ew.virial, eu.virial) {
            (Some(a), Some(b)) => {
                for k in 0..3 {
                    for m in 0..3 {
                        assert!(close(a[k][m], b[k][m], 1e-12));
                    }
                }
            }
            (None, None) => assert_eq!(w.n_beads(), 2),
            other => panic!("virial presence differs: {other:?}"),
        }
    }
}

/// P-5: the P = 1 virial is symmetric (W_ab = W_ba to 1e-13 of the largest entry) on a random
/// periodic configuration; for P = 2 and P = 4 it is `None`.
#[test]
fn virial_symmetric_and_only_for_p1() {
    let cell = Cell::new(RL).unwrap();
    let mut rng = SplitMix64(61);
    let x = random_config(&mut rng, 80, RL, 0.9 * R_SIG, true);
    let w = eval(&mut random_lj(false), &x, Some(&cell)).virial.unwrap();
    let wmax = w.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    for a in 0..3 {
        for b in 0..3 {
            assert!((w[a][b] - w[b][a]).abs() <= 1e-13 * wmax, "{w:?}");
        }
    }
    for p in [2, 4] {
        let q = beads(&vec![x.clone(); p]);
        let ev = random_lj(false).evaluate(&q, Some(&cell), None).unwrap();
        assert_eq!(ev.virial, None, "P = {p}");
        assert_eq!(ev.energies.len(), p);
    }
}

/// C-4: with a cell, cutoff + skin > L_min/2 is an error, for the short axis in each direction
/// and for P = 1 and P = 2, with `PotentialError::InvalidParameter` (Q7); cutoff + skin = L_min/2
/// exactly is accepted. Open boundaries accept any cutoff.
#[test]
fn cell_too_small_is_rejected() {
    // r_c + skin = 3.35: L_min/2 must be ≥ 3.35.
    for axis in 0..3 {
        let mut l = [10.0; 3];
        l[axis] = 6.6; // L/2 = 3.3 < 3.35
        let cell = Cell::new(l).unwrap();
        for p in [1, 2] {
            let q = Beads::zeros(p, 2);
            let mut q = q;
            for j in 0..p {
                q.set(j, 1, [1.5, 1.5, 1.5]);
            }
            let r = random_lj(false).evaluate(&q, Some(&cell), None);
            match r {
                Err(PotentialError::InvalidParameter { .. }) => {}
                Err(e) => panic!("axis {axis}, P = {p}: wrong error kind {e:?} (Q7)"),
                Ok(_) => panic!("axis {axis}, P = {p}: cutoff + skin > L_min/2 accepted (C-4)"),
            }
        }
        l[axis] = 6.7; // L/2 = 3.35 = r_c + skin exactly
        let cell = Cell::new(l).unwrap();
        let x = [[0.0; 3], [1.5, 1.5, 1.5]];
        assert!(
            random_lj(false)
                .evaluate(&beads1(&x), Some(&cell), None)
                .is_ok(),
            "axis {axis}: cutoff + skin = L_min/2 must be accepted (C-4 is ≤)"
        );
    }
    let x = [[0.0; 3], [1.5, 1.5, 1.5]];
    let mut big = LennardJones::new(params(1.0, 1.0, 50.0, 1.0, false)).unwrap();
    assert!(big.evaluate(&beads1(&x), None, None).is_ok());
}

/// P-3: ε, σ and the cutoff must be positive and finite, the skin non-negative and finite;
/// otherwise `PotentialError::InvalidParameter`. skin = 0 and `shift` either way are accepted.
#[test]
fn new_validates_parameters() {
    let bad = [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for v in bad {
        for which in 0..3 {
            let mut p = params(1.0, 1.0, 2.5, 0.3, false);
            match which {
                0 => p.epsilon = v,
                1 => p.sigma = v,
                _ => p.cutoff = v,
            }
            assert!(
                matches!(
                    LennardJones::new(p).err(),
                    Some(PotentialError::InvalidParameter { .. })
                ),
                "parameter {which} = {v} accepted"
            );
        }
    }
    for v in [-1e-12, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let p = params(1.0, 1.0, 2.5, v, false);
        assert!(
            matches!(
                LennardJones::new(p).err(),
                Some(PotentialError::InvalidParameter { .. })
            ),
            "skin = {v} accepted"
        );
    }
    assert!(LennardJones::new(params(1.0, 1.0, 2.5, 0.0, false)).is_ok());
    assert!(LennardJones::new(params(1e-3, 6.4, 16.0, 2.0, true)).is_ok());
}

/// P-4: `subset = Some(..)` returns `PotentialError::Unsupported`, for empty and non-empty subsets,
/// with and without a cell.
#[test]
fn subset_is_unsupported() {
    let cell = Cell::new([12.0; 3]).unwrap();
    let x = [[0.0; 3], [3.0, 0.0, 0.0]];
    for c in [None, Some(&cell)] {
        for subset in [&[][..], &[0][..], &[0, 1][..]] {
            let r = lj(false).evaluate(&beads1(&x), c, Some(subset));
            assert!(
                matches!(r, Err(PotentialError::Unsupported)),
                "subset {subset:?}: {:?}",
                r.err()
            );
        }
    }
}

/// S-3, P-3: evaluating is a pure function of the positions: two calls on the same object with
/// the same positions, and a third after evaluating other positions in between, give identical
/// energies and forces (bitwise; the per-atom accumulation order is fixed, D-3).
#[test]
fn repeated_evaluation_is_identical() {
    let cell = Cell::new(RL).unwrap();
    let mut rng = SplitMix64(71);
    let x = random_config(&mut rng, 60, RL, 0.9 * R_SIG, true);
    let y: Vec<[f64; 3]> = x.iter().map(|r| r.map(|v| v + 0.01)).collect();
    let mut pot = random_lj(false);
    let a = eval(&mut pot, &x, Some(&cell));
    let b = eval(&mut pot, &x, Some(&cell));
    let _ = eval(&mut pot, &y, Some(&cell));
    let c = eval(&mut pot, &x, Some(&cell));
    for e in [&b, &c] {
        assert_eq!(a.energies[0].to_bits(), e.energies[0].to_bits());
        assert_eq!(a.forces, e.forces);
    }
}
