//! Link settings for libtorch from the pixi `ml` environment and the deterministic-algorithms
//! C++ file (MILESTONE_6 "Build isolation", decision 8; spike question 1). Does nothing unless the
//! feature `torch` is enabled, so the default build never needs libtorch.
//!
//! 1. pixi (like conda) sets no `LD_LIBRARY_PATH`; libtorch and its MKL/CUDA dependencies sit in
//!    `$CONDA_PREFIX/lib`, baked in as an rpath. The binaries are tied to the environment's
//!    absolute path; moving the repository needs a rebuild.
//! 2. `libtorch_cuda.so` is reached only through static initialisers, so `--as-needed` drops it
//!    and CUDA is not found; `--no-as-needed` keeps it.
//! 3. `src/deterministic.cpp` is compiled with the system `g++` through `std::process::Command`
//!    against the environment's ATen headers and linked as a static library.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/deterministic.cpp");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if std::env::var_os("CARGO_FEATURE_TORCH").is_none() {
        return;
    }
    let prefix = std::env::var("CONDA_PREFIX")
        .expect("feature `torch`: build inside `pixi run -e ml` (CONDA_PREFIX is not set)");
    let lib = format!("{prefix}/lib");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda,--as-needed");

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let obj = format!("{out}/deterministic.o");
    let status = Command::new("g++")
        .args([
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-c",
        ])
        .arg("src/deterministic.cpp")
        .arg(format!("-I{prefix}/include"))
        .arg(format!("-I{prefix}/include/torch/csrc/api/include"))
        .args(["-o", &obj])
        .status()
        .expect("cannot run g++");
    assert!(status.success(), "src/deterministic.cpp failed to compile");
    let status = Command::new("ar")
        .args(["crs", &format!("{out}/libpimd_torch_det.a"), &obj])
        .status()
        .expect("cannot run ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=pimd_torch_det");
}
