//! Same link settings as crates/pimd-cli/build.rs (rpath $CONDA_PREFIX/lib, --no-as-needed for
//! libtorch_cuda): `cargo:rustc-link-arg` does not propagate from pimd-torch. Build inside
//! `pixi run -e ml`.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    let prefix = std::env::var("CONDA_PREFIX").expect("build inside `pixi run -e ml`");
    let lib = format!("{prefix}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda,--as-needed");
}
