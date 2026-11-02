//! libtorch process-wide settings made by `MacePotential::load` (MILESTONE_6 "Public API";
//! SPEC P-6: the TorchScript profiling executor is disabled and the intra-op thread count fixed at
//! load; D-3 torch note).
//!
//! The settings are process-wide, so this file holds a single test. It reads them back through
//! `tch`, the crate's own backend dependency.
#![cfg(feature = "torch")]

mod common;

use common::load;
use pimd_torch::{Device, Dtype};

/// P-6, D-3: after `load(.., threads, ..)` libtorch's intra-op thread count is `threads` (3, then
/// 2, then 1) and the TorchScript profiling mode is off, even if it was switched on before.
#[test]
fn load_sets_threads_and_disables_profiling() {
    for threads in [3, 2, 1] {
        tch::jit::set_profiling_mode(true);
        let _pot = load(&[8, 1, 1], Device::Cpu, Dtype::F64, threads);
        assert_eq!(
            tch::get_num_threads(),
            threads as i32,
            "intra-op thread count"
        );
        assert!(
            !tch::jit::get_profiling_mode(),
            "the TorchScript profiling executor must be off after load"
        );
    }
}
