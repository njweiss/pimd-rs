// libtorch's deterministic-algorithms switch, which tch 0.26 does not bind (MILESTONE_6
// decision 8). Compiled by build.rs with g++ (no extra crate) and linked statically into
// pimd-torch; called only from the `ffi` module.
#include <ATen/Context.h>

extern "C" void pimd_torch_set_deterministic(int on) {
  at::globalContext().setDeterministicAlgorithms(on != 0, /*warn_only=*/false);
}
