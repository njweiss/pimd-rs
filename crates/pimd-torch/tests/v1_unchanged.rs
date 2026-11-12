//! v1 exports are unchanged (MILESTONE_9 "M9a: Required tests" 3, decision 2; SPEC P-6: "a v1
//! export's energy is used as returned").
//!
//! The locked v1 export `tiny_mace_f64.pt` (no `interface_version()`) must give the same bits as
//! on main. The values below were recorded with main's adapter (commit aa3fe6d, pimd-torch
//! sources identical on milestone-9 at a59a14e), on the owner's workstation (libtorch of the pixi
//! `ml` environment, torch 2.13.0), CPU, libtorch threads 1, RAYON_NUM_THREADS 1 and 4 (same
//! bits): energies as f64 bit patterns per bead, forces as the 64-bit FNV-1a of the little-endian
//! f64 bit patterns of the `[P, N, 3]` array. SPEC D-3 promises bitwise results for a fixed
//! libtorch build, thread count and machine only; on another CPU or libtorch build these values
//! must be re-recorded with main's code (not with the code under test).
#![cfg(feature = "torch")]

mod common;
mod v2_common;

use common::{eval, load, system};
use pimd_torch::Dtype;
use v2_common::fnv1a64_bits;

/// (system, dtype, chunk size, energy bits per bead, FNV-1a of the force bits).
type Case = (&'static str, Dtype, Option<usize>, [u64; 8], u64);

/// The cases, recorded with main's code.
const MAIN: [Case; 6] = [
    (
        "water",
        Dtype::F64,
        None,
        [
            0xc053058cef890a1f,
            0xc053058b38f518ca,
            0xc0530589370c0ae6,
            0xc053058cd5b852a2,
            0xc053059001b9a3b6,
            0xc053058d15fbb34e,
            0xc053058cad3ab80e,
            0xc0530591d3319b58,
        ],
        0xad965af244ef9157,
    ),
    (
        "ethanol",
        Dtype::F64,
        None,
        [
            0xc063370c12921d85,
            0xc063370dd938907a,
            0xc0633706adcc572e,
            0xc06337025b964924,
            0xc063371912c85e5d,
            0xc0633715a313f977,
            0xc0633708a39901c5,
            0xc0633713c82ca85d,
        ],
        0xb65616424567aa7a,
    ),
    (
        "waterbox",
        Dtype::F64,
        None,
        [
            0xc07305890edb7f96,
            0xc07305861388ad0c,
            0xc0730589631ca0a0,
            0xc073058920f70019,
            0xc0730588be8f1340,
            0xc0730588a0bbcea7,
            0xc073058483029b4a,
            0xc073058822ad5a19,
        ],
        0x5b6d930ba9c92b13,
    ),
    (
        "waterbox",
        Dtype::F64,
        Some(3),
        [
            0xc07305890edb7f96,
            0xc07305861388ad0c,
            0xc0730589631ca0a0,
            0xc073058920f70019,
            0xc0730588be8f1340,
            0xc0730588a0bbcea7,
            0xc073058483029b4a,
            0xc073058822ad5a19,
        ],
        0xcf62b089b7c170cf,
    ),
    (
        "water",
        Dtype::F32,
        None,
        [
            0xc053058d1164b47c,
            0xc053058b4dd15551,
            0xc0530589649c58e3,
            0xc053058cebc31738,
            0xc053059027a69b07,
            0xc053058d370651c0,
            0xc053058cc62179f5,
            0xc0530591eb39fa32,
        ],
        0xdb1f04daf191a640,
    ),
    (
        "waterbox",
        Dtype::F32,
        Some(3),
        [
            0xc073058919591e5b,
            0xc0730586031737d0,
            0xc0730589649c58e3,
            0xc073058919591e5b,
            0xc0730588a8744691,
            0xc0730588a8744691,
            0xc07305848ac7132c,
            0xc073058811edd182,
        ],
        0xd9003ad41917b752,
    ),
];

/// P-6 (v1 unchanged, decision 2): the locked v1 export gives main's bits for water, ethanol and
/// the 4-water box (P = 8), unchunked and c = 3, float64 and float32, libtorch threads 1.
#[test]
fn v1_export_gives_mains_bits() {
    for (name, dtype, c, e_bits, f_hash) in MAIN {
        let what = format!("{name}, {dtype:?}, c = {c:?}");
        let s = system(name);
        let mut pot = load(&s.z, pimd_torch::Device::Cpu, dtype, 1);
        if let Some(c) = c {
            pot = pot.with_beads_per_graph(c).unwrap();
        }
        let got = eval(&mut pot, &s.q, s.cell.as_ref());
        let bits: Vec<u64> = got.energies.iter().map(|e| e.to_bits()).collect();
        assert_eq!(bits, e_bits, "{what}: energy bits differ from main");
        assert_eq!(
            fnv1a64_bits(got.forces.as_slice()),
            f_hash,
            "{what}: force bits differ from main"
        );
    }
}
