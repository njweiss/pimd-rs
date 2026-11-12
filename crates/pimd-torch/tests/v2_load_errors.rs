//! v2 load errors (MILESTONE_9 "M9a: Required tests" 4; SPEC P-6: "any libtorch error … →
//! `Backend`", "`load` validates … before it changes libtorch's process-wide settings").
//!
//! Brief: at `load`, a v2 export's `atomic_energies()` must have the length of
//! `atomic_numbers()` and finite values, else `Backend`, and nothing is changed (validate first,
//! settings last); `interface_version()` = 3 is `Backend`. The exports are the stubs of
//! `data/gen_tiny_mace_v2.py` (export interface without MACE; `stub_v2_ok` is the valid control,
//! the others differ from it in one point each):
//! - `len`: 3 E0 values for 4 species; `nan`: a NaN in the table; `inf`: −∞ in the table;
//! - `ver3`: `interface_version()` = 3; `noe0`: `interface_version()` = 2 without
//!   `atomic_energies()` (assumption: a v2 export that cannot give its table is not an export,
//!   `Backend`).
//!
//! The settings are process-wide, so this file holds a single test (as settings_failed_load.rs).
#![cfg(feature = "torch")]

mod common;
mod v2_common;

use common::cuda_available;
use pimd_potentials::PotentialError;
use pimd_torch::{Device, Dtype, MacePotential};
use tch::{Kind, Tensor};
use v2_common::stub;

/// Whether a CUDA op without a deterministic implementation is refused (flag on), as in cuda.rs.
fn deterministic_flag_on() -> bool {
    let x = Tensor::rand([64], (Kind::Float, tch::Device::Cuda(0)));
    x.f_histc(8).is_err()
}

/// P-6 (v2 sentence): the valid stub loads (threads 3, deterministic off; profiling then switched
/// on by the test); each defective stub, with `Dtype::F64` and `Dtype::F32`, is `Backend` at load
/// and leaves the thread count, the profiling mode and (with CUDA) the deterministic switch as
/// they were.
#[test]
fn bad_v2_exports_are_backend_and_change_nothing() {
    let cuda = cuda_available();
    let z = [8u8, 1, 1];
    let _ok = MacePotential::load(&stub("ok"), &z, Device::Cpu, Dtype::F64, 3, false)
        .expect("the valid v2 stub must load (control)");
    tch::jit::set_profiling_mode(true);
    assert_eq!(tch::get_num_threads(), 3, "test setup");
    if cuda {
        assert!(!deterministic_flag_on(), "test setup: deterministic off");
    }
    for name in ["len", "nan", "inf", "ver3", "noe0"] {
        for dtype in [Dtype::F64, Dtype::F32] {
            let what = format!("stub_v2_{name}, {dtype:?}");
            match MacePotential::load(&stub(name), &z, Device::Cpu, dtype, 1, true) {
                Err(PotentialError::Backend(msg)) => {
                    assert!(!msg.is_empty(), "{what}: empty message")
                }
                Err(e) => panic!("{what}: expected Backend, got {e:?}"),
                Ok(_) => panic!("{what}: loaded"),
            }
            assert_eq!(tch::get_num_threads(), 3, "{what}: thread count changed");
            assert!(
                tch::jit::get_profiling_mode(),
                "{what}: profiling mode changed"
            );
            if cuda {
                assert!(
                    !deterministic_flag_on(),
                    "{what}: deterministic switch changed"
                );
            }
        }
    }
}
