//! The one foreign function of the workspace: libtorch's deterministic-algorithms switch, which
//! tch 0.26 does not bind, from `src/deterministic.cpp` (compiled by build.rs; MILESTONE_6
//! decision 8).

// SAFETY: `pimd_torch_set_deterministic` is defined in src/deterministic.cpp with exactly this C
// signature (`void(int)`), takes no pointers, and is `noexcept` with a catch-all, so no C++
// exception unwinds into Rust. It writes libtorch's process-wide deterministic-algorithms flag, a
// plain bool: calls concurrent with each other or with libtorch ops on other threads are a formal
// data race (as with PyTorch's own `torch.use_deterministic_algorithms`). pimd-torch calls it
// only from `MacePotential::load`.
unsafe extern "C" {
    pub(crate) safe fn pimd_torch_set_deterministic(on: i32);
}
