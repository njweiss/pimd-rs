//! Edges under periodic boundaries (MILESTONE_6 "Edges"; SPEC P-6, C-2, C-5).
//!
//! P-6: edges are the ordered pairs i ≠ j of one bead with |q_j − q_i + s| < r_max, s the C-2
//! minimum-image correction; r_max < L_min/2 is required, else `InvalidParameter { what:
//! "cutoff", value: r_max }` (bohr). The correctness of the boundary-crossing edges against mace's
//! own neighbour list is checked in `reference.rs` (4-water box); here: invariance under wrapping
//! (C-5: ring-polymer positions are unwrapped, potentials apply the minimum image), and the cutoff
//! condition. float64, CPU, 1 libtorch thread; tolerance 1e-12 relative (brief).
#![cfg(feature = "torch")]

mod common;

use common::{BOHR_A, errors, eval, load, system};
use pimd_core::{Beads, Cell};
use pimd_potentials::{Potential, PotentialError};
use pimd_torch::{Device, Dtype};

/// The test's own wrap into [0, L): x − L·floor(x/L), per component.
fn wrap(x: [f64; 3], l: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| x[k] - l[k] * (x[k] / l[k]).floor())
}

/// P-6, C-2, C-5: the 4-water box (8 beads, positions unwrapped as stored) gives the same
/// energies and forces (1e-12 relative) with (a) every atom wrapped into [0, L) and (b) every atom
/// shifted by its own whole number of box lengths in −2 … 2 per axis (images far outside the box).
#[test]
fn wrapped_and_shifted_positions_give_the_same_result() {
    let s = system("waterbox");
    let cell = s.cell.expect("periodic");
    let l = cell.lengths();
    let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
    let unwrapped = eval(&mut pot, &s.q, Some(&cell));
    let mut wrapped = Beads::zeros(s.p, s.n);
    let mut shifted = Beads::zeros(s.p, s.n);
    let mut rng = common::SplitMix64(17);
    let mut moved = 0;
    for j in 0..s.p {
        for i in 0..s.n {
            let x = s.q.get(j, i);
            let w = wrap(x, l);
            if w != x {
                moved += 1;
            }
            wrapped.set(j, i, w);
            let n: [f64; 3] = [0, 1, 2].map(|_| (rng.next_f64() * 5.0).floor() - 2.0);
            shifted.set(j, i, [0, 1, 2].map(|k| x[k] + n[k] * l[k]));
        }
    }
    assert!(
        moved > 0,
        "the stored box positions must include unwrapped atoms"
    );
    for (what, q) in [("wrapped", &wrapped), ("shifted", &shifted)] {
        let got = eval(&mut pot, q, Some(&cell));
        let (er, _, fr) = errors(&got, &unwrapped.energies, &unwrapped.forces);
        eprintln!("{what}: |dE|/|E| {er:.2e}, |dF|/max|F| {fr:.2e}");
        assert!(er <= 1e-12, "{what}: energy {er:e}");
        assert!(fr <= 1e-12, "{what}: forces {fr:e}");
    }
}

/// P-6, C-1: a molecule far from the faces of a large box (100 Å, so no pair is near a boundary)
/// gives the gas-phase result (1e-12 relative).
#[test]
fn large_box_equals_gas_phase() {
    let s = system("water");
    let l = 100.0 / BOHR_A;
    let cell = Cell::new([l; 3]).unwrap();
    let mut q = s.q.clone();
    for j in 0..s.p {
        for i in 0..s.n {
            q.set(j, i, q.get(j, i).map(|c| c + 0.5 * l));
        }
    }
    let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
    let gas = eval(&mut pot, &s.q, None);
    let boxed = eval(&mut pot, &q, Some(&cell));
    let (er, _, fr) = errors(&boxed, &gas.energies, &gas.forces);
    assert!(er <= 1e-12, "energy {er:e}");
    assert!(fr <= 1e-12, "forces {fr:e}");
}

/// P-6: with a cell, r_max < L_min/2 is required. r_max = 5 Å; boxes (Å) 9.99³ and 20×20×9.99
/// (L_min/2 = 4.995 Å ≤ r_max) give `InvalidParameter { what: "cutoff", value: r_max }` with r_max
/// in bohr (5 / 0.529177210903, 1e-15); 10.01³ (L_min/2 = 5.005 Å) evaluates.
#[test]
fn cutoff_at_least_half_box_is_rejected() {
    let s = system("water_single");
    let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
    let r = 5.0 / BOHR_A;
    for lengths in [[9.99; 3], [20.0, 20.0, 9.99], [9.99, 20.0, 20.0]] {
        let cell = Cell::new(lengths.map(|a| a / BOHR_A)).unwrap();
        match pot.evaluate(&s.q, Some(&cell), None) {
            Err(PotentialError::InvalidParameter { what, value }) => {
                assert_eq!(what, "cutoff");
                assert!(
                    (value - r).abs() <= 1e-15 * r,
                    "value {value}, want r_max = {r} bohr"
                );
            }
            other => panic!("box {lengths:?} Å: expected InvalidParameter(cutoff), got {other:?}"),
        }
    }
    let ok = Cell::new([10.01 / BOHR_A; 3]).unwrap();
    pot.evaluate(&s.q, Some(&ok), None)
        .expect("10.01 Å box: r_max < L/2, must evaluate");
}
