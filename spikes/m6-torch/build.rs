//! Link settings for libtorch from the pixi `ml` environment (spike question 1), and the small C++
//! shim for `at::globalContext().setDeterministicAlgorithms` (question 6).
//!
//! 1. `libtorch_cuda.so` is only reached through static initialisers, so `--as-needed` drops it
//!    and `tch::Cuda::is_available()` is false. Force it in with `--no-as-needed`.
//! 2. pixi (like conda) does not set `LD_LIBRARY_PATH`; libtorch and its MKL/CUDA dependencies sit
//!    in `$CONDA_PREFIX/lib`. Bake that directory in as an rpath so the binary runs without
//!    `pixi run` and without `LD_LIBRARY_PATH`.
//! 3. `shim.cpp` is compiled with the system `g++` through `std::process::Command` (no `cc` crate)
//!    against the environment's ATen headers.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-changed=shim.cpp");
    let prefix = std::env::var("CONDA_PREFIX").expect("build inside `pixi run -e ml`");
    let lib = format!("{prefix}/lib");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda,--as-needed");

    let out = std::env::var("OUT_DIR").unwrap();
    let obj = format!("{out}/shim.o");
    let st = Command::new("g++")
        .args(["-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1", "-c", "shim.cpp"])
        .arg(format!("-I{prefix}/include"))
        .arg(format!("-I{prefix}/include/torch/csrc/api/include"))
        .args(["-o", &obj])
        .status()
        .expect("g++");
    assert!(st.success(), "shim.cpp failed to compile");
    let st = Command::new("ar")
        .args(["crs", &format!("{out}/libspikeshim.a"), &obj])
        .status()
        .expect("ar");
    assert!(st.success());
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=spikeshim");
}
