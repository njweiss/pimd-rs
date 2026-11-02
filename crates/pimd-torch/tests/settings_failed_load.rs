//! A failed `load` leaves libtorch's process-wide settings alone (MILESTONE_6 "After review", N2;
//! SPEC P-6, D-3 torch note).
//!
//! `load` validates the file, the module and the atomic numbers before it sets the thread count,
//! the profiling mode and the deterministic switch, so a load that fails changes none of them.
//! The settings are process-wide, so this file holds a single test.
#![cfg(feature = "torch")]

mod common;

use common::{cuda_available, export_path, pickled_model_path};
use pimd_torch::{Device, Dtype, MacePotential};
use tch::{Kind, Tensor};

/// Whether a CUDA op without a deterministic implementation is refused (flag on), as in cuda.rs.
fn deterministic_flag_on() -> bool {
    let x = Tensor::rand([64], (Kind::Float, tch::Device::Cuda(0)));
    x.f_histc(8).is_err()
}

/// P-6, D-3: start from threads = 3, deterministic off (a successful load) and profiling on; then
/// loads that fail (unknown element, missing file, a pickled MACE model instead of an export, and
/// with CUDA an out-of-range device index) leave all three as they were.
#[test]
fn failed_load_leaves_settings_unchanged() {
    let cuda = cuda_available();
    let _ok = MacePotential::load(
        &export_path(),
        &[8, 1, 1],
        Device::Cpu,
        Dtype::F64,
        3,
        false,
    )
    .expect("baseline load");
    tch::jit::set_profiling_mode(true);
    assert_eq!(tch::get_num_threads(), 3, "test setup");
    if cuda {
        assert!(!deterministic_flag_on(), "test setup: deterministic off");
    }

    let missing = export_path().with_file_name("no_such_export.pt");
    let mut cases: Vec<(&str, std::path::PathBuf, Vec<u8>, Device)> = vec![
        (
            "unknown element (Fe)",
            export_path(),
            vec![8, 26, 1],
            Device::Cpu,
        ),
        ("missing file", missing, vec![8, 1, 1], Device::Cpu),
        (
            "pickled model, not an export",
            pickled_model_path(),
            vec![8, 1, 1],
            Device::Cpu,
        ),
    ];
    if cuda {
        let n = usize::try_from(tch::Cuda::device_count()).expect("device count");
        cases.push((
            "CUDA index out of range",
            export_path(),
            vec![8, 1, 1],
            Device::Cuda(n),
        ));
    }
    for (name, path, z, device) in cases {
        let r = MacePotential::load(&path, &z, device, Dtype::F64, 1, true);
        assert!(r.is_err(), "{name}: load must fail");
        assert_eq!(tch::get_num_threads(), 3, "{name}: thread count changed");
        assert!(
            tch::jit::get_profiling_mode(),
            "{name}: profiling mode changed"
        );
        if cuda {
            assert!(
                !deterministic_flag_on(),
                "{name}: deterministic switch changed"
            );
        }
    }
}
