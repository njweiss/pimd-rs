//! The one foreign function of the workspace: libtorch's deterministic-algorithms switch, which
//! tch 0.26 does not bind, from `src/deterministic.cpp` (compiled by build.rs; MILESTONE_6
//! decision 8).

// SAFETY: `pimd_torch_set_deterministic` is defined in src/deterministic.cpp with exactly this C
// signature (`void(int)`), takes no pointers, and only flips libtorch's process-wide
// deterministic-algorithms flag; it is safe to call from any thread at any time.
unsafe extern "C" {
    #[allow(dead_code)] // stub: called by `MacePotential::load` in M6b
    pub(crate) safe fn pimd_torch_set_deterministic(on: i32);
}
