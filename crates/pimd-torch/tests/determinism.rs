//! Bitwise repeatability on CPU (MILESTONE_6 "Determinism"; SPEC D-3 torch note, P-6).
//!
//! D-3: on CPU, results for a fixed libtorch thread count are bitwise reproducible, including the
//! first call after a (re)load, because the adapter disables the TorchScript profiling executor
//! (P-6). With the profiling executor on, the first call runs the profiled graph and later calls an
//! optimised one that rounds differently (spike Q6: first call ≠ calls 2 … 20 in every case
//! measured, including this tiny model).
//!
//! The thread count is a process-wide libtorch setting, so this file holds a single test that
//! runs its cases one after another (cargo runs the tests of one file in parallel threads).
#![cfg(feature = "torch")]

mod common;

use common::{bitwise_eq, displaced, eval, load, max_diff, system};
use pimd_torch::{Device, Dtype};

/// D-3, P-6: for threads 1 and 4, float64 and float32, ethanol (8 beads) and water (16 beads, the
/// M5 geometry displaced by up to ±0.1 bohr): the first and second calls are bitwise equal; 20
/// calls are bitwise equal; the first call of a freshly loaded module is bitwise equal to them
/// (restart: every call runs the same graph).
#[test]
fn cpu_calls_are_bitwise_repeatable() {
    let eth = system("ethanol");
    let wat = system("water");
    let wat16 = displaced(&wat.base, 16, 0.1, 31);
    let cases = [
        ("ethanol P = 8", &eth.z, &eth.q),
        ("water P = 16", &wat.z, &wat16),
    ];
    for threads in [1, 4] {
        for dtype in [Dtype::F64, Dtype::F32] {
            for (name, z, q) in cases {
                let what = format!("{name}, {dtype:?}, {threads} thread(s)");
                let mut pot = load(z, Device::Cpu, dtype, threads);
                let first = eval(&mut pot, q, None);
                for call in 2..=20 {
                    let next = eval(&mut pot, q, None);
                    let (de, df) = max_diff(&next, &first);
                    assert!(
                        bitwise_eq(&next, &first),
                        "{what}: call {call} differs from call 1 (|dE| {de:e}, |dF| {df:e}): \
                         profiling executor left on?"
                    );
                }
                let mut again = load(z, Device::Cpu, dtype, threads);
                let fresh = eval(&mut again, q, None);
                assert!(
                    bitwise_eq(&fresh, &first),
                    "{what}: first call after reloading differs"
                );
            }
        }
    }
}
