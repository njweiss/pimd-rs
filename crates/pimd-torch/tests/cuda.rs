//! CUDA rows (MILESTONE_6 "Per evaluation vs Python MACE" CUDA rows, "Determinism"; SPEC P-6,
//! D-1, D-3 torch note). Skipped with a message when libtorch sees no CUDA device.
//!
//! The deterministic-algorithms switch is process-wide, so this file holds a single test that
//! runs its parts in order.
//!
//! - Accuracy: as `reference.rs` on `Device::Cuda(0)`: float64 |ΔE| ≤ 1e-12·|E|, max|ΔF| ≤
//!   1e-10·max|F| (spike: ΔE ≤ 2.7e-12 eV ≈ 7e-16 relative, ΔF ≤ 1.6e-15); float32 against the
//!   float64 reference |ΔE| ≤ 5e-3 eV, max|ΔF| ≤ 1e-4·max|F|.
//! - The switch itself: `load(.., deterministic)` sets libtorch's deterministic-algorithms flag
//!   (decision 8). Observed through an op that has no deterministic CUDA implementation
//!   (`histc` on a CUDA float tensor raises an error while the flag is on, PyTorch 2.13).
//! - D-3: with `deterministic = true`, 20 calls on ethanol with P = 32 (288 atoms; every call
//!   differed without the switch in the spike, MACE's scatter sums use atomics) are bitwise equal,
//!   float64 and float32, including the first call.
#![cfg(feature = "torch")]

mod common;

use common::{
    EV_PER_EH, SYSTEMS, bitwise_eq, displaced, errors, eval, export_path, max_diff, system,
};
use pimd_torch::{Device, Dtype, MacePotential};
use tch::{Kind, Tensor};

fn load_cuda(z: &[u8], dtype: Dtype, deterministic: bool) -> MacePotential {
    MacePotential::load(&export_path(), z, Device::Cuda(0), dtype, 1, deterministic)
        .unwrap_or_else(|e| panic!("CUDA load failed: {e:?}"))
}

/// Whether a CUDA op without a deterministic implementation is refused (flag on).
fn deterministic_flag_on() -> bool {
    let x = Tensor::rand([64], (Kind::Float, tch::Device::Cuda(0)));
    x.f_histc(8).is_err()
}

/// P-6, D-1, D-3: CUDA accuracy against Python MACE, the deterministic switch, and bitwise
/// repeatability with `deterministic = true` (ethanol P = 32).
#[test]
fn cuda_accuracy_switch_and_repeatability() {
    if !common::cuda_available() {
        eprintln!("SKIP cuda_accuracy_switch_and_repeatability: libtorch sees no CUDA device");
        return;
    }
    // Accuracy.
    for name in SYSTEMS {
        let s = system(name);
        let got = eval(
            &mut load_cuda(&s.z, Dtype::F64, true),
            &s.q,
            s.cell.as_ref(),
        );
        let (er, _, fr) = errors(&got, &s.energies, &s.forces);
        eprintln!("{name} f64 cuda: |dE|/|E| {er:.2e}, |dF|/max|F| {fr:.2e}");
        assert!(er <= 1e-12, "{name} f64 CUDA: energy {er:e}");
        assert!(fr <= 1e-10, "{name} f64 CUDA: forces {fr:e}");
        let got = eval(
            &mut load_cuda(&s.z, Dtype::F32, true),
            &s.q,
            s.cell.as_ref(),
        );
        let (_, ea, fr) = errors(&got, &s.energies, &s.forces);
        eprintln!(
            "{name} f32 cuda: |dE| {:.2e} eV, |dF|/max|F| {fr:.2e}",
            ea * EV_PER_EH
        );
        assert!(ea <= 5e-3 / EV_PER_EH, "{name} f32 CUDA: energy {ea:e} Eh");
        assert!(fr <= 1e-4, "{name} f32 CUDA: forces {fr:e}");
    }
    // The switch follows `deterministic` at every load.
    let eth = system("ethanol");
    let _off = load_cuda(&eth.z, Dtype::F64, false);
    assert!(
        !deterministic_flag_on(),
        "deterministic = false: flag must be off"
    );
    let _on = load_cuda(&eth.z, Dtype::F64, true);
    assert!(
        deterministic_flag_on(),
        "deterministic = true: flag must be on"
    );
    let _off = load_cuda(&eth.z, Dtype::F32, false);
    assert!(
        !deterministic_flag_on(),
        "deterministic = false: flag must be off"
    );
    // Repeatability with the switch on.
    let q = displaced(&eth.base, 32, 0.1, 32);
    for dtype in [Dtype::F64, Dtype::F32] {
        let mut pot = load_cuda(&eth.z, dtype, true);
        let first = eval(&mut pot, &q, None);
        for call in 2..=20 {
            let next = eval(&mut pot, &q, None);
            let (de, df) = max_diff(&next, &first);
            assert!(
                bitwise_eq(&next, &first),
                "ethanol P = 32 {dtype:?} CUDA deterministic: call {call} differs (|dE| {de:e}, \
                 |dF| {df:e})"
            );
        }
    }
}
