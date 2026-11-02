//! Errors and `atomic_number` (MILESTONE_6 "Errors", "Public API"; SPEC P-6, P-1, P-4).
//!
//! P-6: `subset = Some` → `Unsupported` (P-4); `positions.n_atoms() ≠ N` → `ShapeMismatch` (P-1);
//! `load`: an atomic number the model does not know → `InvalidParameter { what: "atomic number" }`,
//! `threads == 0` → `InvalidParameter { what: "threads" }`, a missing or invalid file →
//! `Backend`, CUDA unavailable → `Backend`.
#![cfg(feature = "torch")]

mod common;

use common::{eval, export_path, load, pickled_model_path, system};
use pimd_core::Beads;
use pimd_potentials::{Potential, PotentialError};
use pimd_torch::{Device, Dtype, MacePotential, atomic_number};

/// P-4, P-6: `subset = Some(..)` is `Unsupported`, for an empty, a partial and a full subset.
#[test]
fn subset_is_unsupported() {
    let s = system("water");
    let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
    for sub in [&[][..], &[0][..], &[0, 1, 2][..]] {
        assert_eq!(
            pot.evaluate(&s.q, None, Some(sub)),
            Err(PotentialError::Unsupported),
            "subset {sub:?}"
        );
    }
    // Still usable afterwards.
    eval(&mut pot, &s.q, None);
}

/// P-1, P-6: positions with N ≠ 3 atoms for a 3-atom potential → `ShapeMismatch { expected: 3,
/// actual }` (actual 2 and 4).
#[test]
fn wrong_atom_count_is_shape_mismatch() {
    let mut pot = load(&[8, 1, 1], Device::Cpu, Dtype::F64, 1);
    for n in [2, 4] {
        let mut q = Beads::zeros(2, n);
        for i in 0..n {
            for j in 0..2 {
                q.set(j, i, [2.0 * i as f64, 0.1 * j as f64, 0.0]);
            }
        }
        assert_eq!(
            pot.evaluate(&q, None, None),
            Err(PotentialError::ShapeMismatch {
                expected: 3,
                actual: n
            })
        );
    }
}

/// P-6: atomic numbers the tiny model does not know (it has H, C, N, O): F (9), 0, 2 (He) and 200
/// → `InvalidParameter { what: "atomic number" }`.
#[test]
fn unknown_atomic_number_is_rejected() {
    for z in [9u8, 0, 2, 200] {
        match MacePotential::load(&export_path(), &[8, 1, z], Device::Cpu, Dtype::F64, 1, true) {
            Err(PotentialError::InvalidParameter { what, .. }) => {
                assert_eq!(what, "atomic number", "Z = {z}")
            }
            Err(e) => panic!("Z = {z}: expected InvalidParameter(atomic number), got {e:?}"),
            Ok(_) => panic!("Z = {z}: accepted an element the model does not know"),
        }
    }
}

/// P-6: `threads = 0` → `InvalidParameter { what: "threads" }`.
#[test]
fn zero_threads_is_rejected() {
    match MacePotential::load(&export_path(), &[8, 1, 1], Device::Cpu, Dtype::F64, 0, true) {
        Err(PotentialError::InvalidParameter { what, .. }) => assert_eq!(what, "threads"),
        Err(e) => panic!("expected InvalidParameter(threads), got {e:?}"),
        Ok(_) => panic!("threads = 0 accepted"),
    }
}

fn expect_backend(path: &std::path::Path, what: &str) {
    match MacePotential::load(path, &[8, 1, 1], Device::Cpu, Dtype::F64, 1, true) {
        Err(PotentialError::Backend(msg)) => assert!(!msg.is_empty(), "{what}: empty message"),
        Err(e) => panic!("{what}: expected Backend, got {e:?}"),
        Ok(_) => panic!("{what}: loaded"),
    }
}

/// P-6: a missing file, an empty file, a text file and a MACE model file that is not a
/// TorchScript export (the committed pickled `tiny_mace.model`) → `Backend`.
#[test]
fn missing_or_invalid_file_is_backend_error() {
    let d = tempfile::tempdir().unwrap();
    expect_backend(&d.path().join("does-not-exist.pt"), "missing file");
    let empty = d.path().join("empty.pt");
    std::fs::write(&empty, b"").unwrap();
    expect_backend(&empty, "empty file");
    let text = d.path().join("text.pt");
    std::fs::write(&text, b"this is not a TorchScript archive\n").unwrap();
    expect_backend(&text, "text file");
    expect_backend(&pickled_model_path(), "pickled MACE model (not an export)");
    expect_backend(d.path(), "a directory");
}

/// P-6: CUDA unavailable → `Backend` (runs only on machines without CUDA).
#[test]
fn cuda_unavailable_is_backend_error() {
    if common::cuda_available() {
        eprintln!("SKIP cuda_unavailable_is_backend_error: CUDA is available on this machine");
        return;
    }
    match MacePotential::load(
        &export_path(),
        &[8, 1, 1],
        Device::Cuda(0),
        Dtype::F64,
        1,
        true,
    ) {
        Err(PotentialError::Backend(_)) => {}
        Err(e) => panic!("expected Backend, got {e:?}"),
        Ok(_) => panic!("loaded on CUDA without CUDA"),
    }
}

/// The element symbols H (1) through I (53), in order (periodic table).
const ELEMENTS: [&str; 53] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I",
];

/// Public API: `atomic_number` maps H … I to 1 … 53 (exact enumeration) and returns `None` for
/// anything else: Xe (54, beyond I), the empty string, lower case, upper-case two-letter forms,
/// surrounding whitespace, isotope labels and numbers.
#[test]
fn atomic_number_h_through_i() {
    for (k, s) in ELEMENTS.iter().enumerate() {
        assert_eq!(atomic_number(s), Some(k as u8 + 1), "{s}");
    }
    for s in [
        "Xe", "Cs", "U", "", "h", "o", "HE", "CL", " H", "H ", "D", "T", "X", "1", "Hx",
    ] {
        assert_eq!(atomic_number(s), None, "{s:?}");
    }
}
