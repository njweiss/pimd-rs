//! With the feature `torch`, the binaries and tests of this crate link libtorch (through
//! `pimd-torch` and `tch`) and need the same link settings as `pimd-torch`'s own targets
//! (`crates/pimd-torch/build.rs`): `cargo:rustc-link-arg` does not propagate to dependent crates.
//! rpath `$CONDA_PREFIX/lib` (pixi sets no `LD_LIBRARY_PATH`) and `--no-as-needed` for
//! `libtorch_cuda.so` (else CUDA is not found). Does nothing without the feature.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if std::env::var_os("CARGO_FEATURE_TORCH").is_none() {
        return;
    }
    let prefix = std::env::var("CONDA_PREFIX")
        .expect("feature `torch`: build inside `pixi run -e ml` (CONDA_PREFIX is not set)");
    let lib = format!("{prefix}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda,--as-needed");
}
