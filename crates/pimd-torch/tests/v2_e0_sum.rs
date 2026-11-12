//! The adapter's E0 sum (MILESTONE_9 "M9a: Adapter"; SPEC P-6 v2 sentence, U-3, D-1, D-3).
//!
//! Brief: E_j = f64(E_int,j)·(eV→Eh) + Σ_i E0(Z_i)·(eV→Eh), the sum in f64 in atom order, read
//! from `atomic_energies()` before any cast to float32, so the result is the same for every bead
//! and chunk.
//!
//! Export: `stub_v2_ok.pt` (`data/gen_tiny_mace_v2.py`): `atomic_numbers()` = [8, 1, 6, 7] (not
//! sorted, so the species index differs from the rank of Z), `atomic_energies()` = the float64
//! table `stub.e0_ev` of `tiny_mace_v2.json` (−2042.123456789, −13.587654321, −1029.314159265,
//! −1485.271828183 eV; not float32-representable), and a `forward` that returns the interaction
//! energy 0.375 eV (exact in float32) for every graph and zero forces.
//!
//! Expected energy (hand derivation): every bead of atoms Z = [8, 1, 1, 6, 7, 1] has
//! E = (0.375 + Σ_i E0(Z_i)) / 27.211386245988 Eh (U-2 literal, Q13), the sum taken here in f64;
//! |E| ≈ 169 Eh. Tolerance 1e-14 relative: the brief fixes the order of the sum but the
//! conversion may be applied per term or to the sum, which differ by a few f64 ulp (≈ 1e-16
//! relative each). A float32 E0 sum or table is off by ≈ 1e-8 relative.
//! libtorch threads 1.
#![cfg(feature = "torch")]

mod common;
mod v2_common;

use common::{EV_PER_EH, eval};
use pimd_core::Beads;
use pimd_torch::Dtype;
use v2_common::{bits_eq, load_path, stub, v2_json};

const Z: [u8; 6] = [8, 1, 1, 6, 7, 1];
const P: usize = 5;

/// Five beads of six atoms, spread so that no two atoms coincide (bohr).
fn beads() -> Beads {
    let mut q = Beads::zeros(P, Z.len());
    for j in 0..P {
        for i in 0..Z.len() {
            q.set(j, i, [2.5 * i as f64, 0.3 * j as f64, 0.1 * (i * j) as f64]);
        }
    }
    q
}

/// (0.375 + Σ_i E0(Z_i)) / 27.211386245988, from the JSON table, in Eh.
fn expected() -> f64 {
    let j = v2_json();
    let st = j.get("stub");
    let zs: Vec<u8> = st
        .get("atomic_numbers")
        .nums()
        .iter()
        .map(|&z| z as u8)
        .collect();
    let e0 = st.get("e0_ev").nums();
    let mut sum = st.get("interaction_energy_ev").num();
    for z in Z {
        sum += e0[zs.iter().position(|&m| m == z).unwrap()];
    }
    sum / EV_PER_EH
}

/// P-6 (v2 sentence), U-3, D-1: with `Dtype::F64` and `Dtype::F32`, every bead's energy is the
/// interaction energy plus the f64 sum of the E0 table, in Eh, to 1e-14 relative; forces are the
/// stub's zeros; the float32 energies are bitwise equal to the float64 ones (the table is read
/// before the cast, and 0.375 is exact in float32).
#[test]
fn energy_is_interaction_plus_f64_e0_sum() {
    let want = expected();
    let q = beads();
    let mut by_dtype = Vec::new();
    for dtype in [Dtype::F64, Dtype::F32] {
        let mut pot = load_path(&stub("ok"), &Z, dtype, 1);
        let got = eval(&mut pot, &q, None);
        assert_eq!(got.energies.len(), P, "{dtype:?}: one energy per bead");
        for (b, e) in got.energies.iter().enumerate() {
            let rel = (e - want).abs() / want.abs();
            assert!(
                rel <= 1e-14,
                "{dtype:?} bead {b}: E {e} Eh, expected {want} Eh ({rel:e} relative)"
            );
        }
        assert!(
            got.forces.as_slice().iter().all(|&f| f == 0.0),
            "{dtype:?}: forces are the stub's zeros"
        );
        by_dtype.push(got.energies);
    }
    assert!(
        bits_eq(&by_dtype[0], &by_dtype[1]),
        "float32 energies {:?} differ from float64 energies {:?}",
        by_dtype[1],
        by_dtype[0]
    );
}

/// D-3, P-6: the E0 sum is the same for every bead and chunk: all P = 5 energies are bitwise
/// equal, unchunked and for c = 1, 2, 3 (chunks 2+2+1 and 3+2 included), and every chunked result
/// is bitwise equal to the unchunked one; repeated calls give the same bits.
#[test]
fn e0_sum_identical_for_every_bead_and_chunk() {
    let q = beads();
    let mut pot = load_path(&stub("ok"), &Z, Dtype::F64, 1);
    let base = eval(&mut pot, &q, None);
    assert!(
        base.energies
            .iter()
            .all(|e| e.to_bits() == base.energies[0].to_bits()),
        "unchunked: bead energies differ: {:?}",
        base.energies
    );
    let again = eval(&mut pot, &q, None);
    assert!(
        bits_eq(&again.energies, &base.energies),
        "second call differs"
    );
    for c in [1, 2, 3] {
        let mut pot = load_path(&stub("ok"), &Z, Dtype::F64, 1)
            .with_beads_per_graph(c)
            .unwrap();
        let got = eval(&mut pot, &q, None);
        assert!(
            bits_eq(&got.energies, &base.energies),
            "c = {c}: {:?} differ from unchunked {:?}",
            got.energies,
            base.energies
        );
    }
}
