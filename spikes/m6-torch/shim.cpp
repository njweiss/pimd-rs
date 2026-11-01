// Two libtorch global switches that tch 0.26 does not expose (spike question 6). Compiled by
// build.rs with g++ (no extra crate) and linked statically into the spike binary.
#include <ATen/Context.h>

extern "C" void spike_set_deterministic(int on) {
  at::globalContext().setDeterministicAlgorithms(on != 0, /*warn_only=*/false);
}

extern "C" int spike_get_deterministic() {
  return at::globalContext().deterministicAlgorithms() ? 1 : 0;
}
