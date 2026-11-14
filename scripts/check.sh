#!/usr/bin/env bash
# Single pass/fail gate. Exits non-zero on the first failure.
#   ./scripts/check.sh              fast checks (what implementers iterate on)
#   CHECK_LONG=1 ./scripts/check.sh also runs long validation and reference-data checks
#   CHECK_IPI=1 (with CHECK_LONG=1) also runs the socket validation (water on GFN2-xTB, feature
#               socket-validation, about 1 h) and regenerates the slow i-PI references
#               (neon about 34 min, water/xTB about 1 h)
#   CHECK_TORCH=1 also runs the in-process torch backend (pimd-torch, pimd-cli with feature `torch`)
#               inside `pixi run -e ml`; with CHECK_LONG=1 also its long validation (MACE-OFF23 via
#               PIMD_MACE_MODEL, skipped with a message when unset)
#   CHECK_CUEQ=1 also runs the MACE socket client with cuEquivariance kernels (SPEC K-7; needs a GPU
#               and the cueq venv out of the repo, README): CUEQ_PYTHON names its python (default
#               .venv-cueq/bin/python). With CHECK_TORCH=1 the torch steps run those tests; otherwise
#               a separate step does (with CHECK_LONG=1 also the long client test)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

TESTS_GLOB=':(glob)crates/*/tests/**'

if [ "${CHECK_CUEQ:-0}" = "1" ]; then
  # MILESTONE_10: the cueq packages never enter pixi.toml/pixi.lock (owner decision 2); the tests
  # start the client with this interpreter inside `pixi run -e ml` (Triton needs its ptxas).
  export CUEQ_PYTHON="${CUEQ_PYTHON:-$PWD/.venv-cueq/bin/python}"
  [ -x "$CUEQ_PYTHON" ] || fail "CHECK_CUEQ=1: no cueq venv python at $CUEQ_PYTHON (README; or set CUEQ_PYTHON)"
fi

step "formatting"
cargo fmt --all -- --check

step "clippy"
cargo clippy --workspace --all-targets -- -D warnings

step "no ignored tests"
if grep -rnE '#\[ignore' crates/*/tests 2>/dev/null; then
  fail "#[ignore] found in tests"
fi

step "test lock"
git_dir="$(git rev-parse --git-dir)"
if [ -f "$git_dir/tests-lock" ]; then
  lock="$(tr -d '[:space:]' < "$git_dir/tests-lock")"
  [ -n "$lock" ] || fail "tests-lock is empty; run /implement again"
  git cat-file -e "${lock}^{commit}" 2>/dev/null || fail "tests-lock does not name a commit"
  if ! git diff --quiet "$lock" -- "$TESTS_GLOB"; then
    git diff --stat "$lock" -- "$TESTS_GLOB" >&2
    fail "tests, reference data or generators differ from the locked test commit ${lock:0:12}"
  fi
  untracked="$(git ls-files --others --exclude-standard -- "$TESTS_GLOB")"
  [ -z "$untracked" ] || { printf '%s\n' "$untracked" >&2; fail "new untracked files under crates/*/tests"; }
  echo "tests unchanged since ${lock:0:12}"
else
  echo "no tests-lock (test-authoring phase): skipping"
fi

step "cargo test, 1 thread (determinism: results must not depend on thread count)"
RAYON_NUM_THREADS=1 cargo test --workspace

step "cargo test, 4 threads"
RAYON_NUM_THREADS=4 cargo test --workspace

if [ "${CHECK_TORCH:-0}" = "1" ]; then
  # MILESTONE_6 "CHECK_TORCH mechanics": libtorch comes from the pixi `ml` environment; the default
  # steps above never build it (pimd-torch is empty without its feature).
  step "torch backend (pixi ml env)"
  pixi run -e ml env LIBTORCH_USE_PYTORCH=1 bash -c '
    cargo clippy -p pimd-torch -p pimd-cli --all-targets --features torch -- -D warnings &&
    RAYON_NUM_THREADS=1 cargo test -p pimd-torch -p pimd-cli --features torch &&
    RAYON_NUM_THREADS=4 cargo test -p pimd-torch -p pimd-cli --features torch'
fi

if [ "${CHECK_LONG:-0}" = "1" ]; then
  # Every crate that has long tests defines the feature `long-validation`; the water/xTB
  # comparison with i-PI (feature `socket-validation`, pimd-cli) runs only with CHECK_IPI=1.
  features="long-validation"
  if [ "${CHECK_IPI:-0}" = "1" ]; then
    features="long-validation,socket-validation"
  fi
  step "long validation (release, features: $features)"
  RAYON_NUM_THREADS=4 cargo test --workspace --release --features "$features"

  if [ "${CHECK_TORCH:-0}" = "1" ]; then
    # pimd-torch, and from pimd-cli only the M8a MACE-OFF23 water vs i-PI test (about 2 h on the
    # GPU): pimd-cli's other long tests (neon) already ran above and have no torch part.
    step "torch long validation (pixi ml env, release, features: torch,long-validation)"
    pixi run -e ml env LIBTORCH_USE_PYTORCH=1 RAYON_NUM_THREADS=4 \
      cargo test -p pimd-torch --release --features torch,long-validation
    # MILESTONE_10 test 7: the MACE socket client on the M8b box (skips without the model file).
    pixi run -e ml env LIBTORCH_USE_PYTORCH=1 RAYON_NUM_THREADS=4 \
      cargo test -p pimd-cli --release --features torch,long-validation --test long_water_mace_ipi \
      --test long_mace_client
  fi

  if [ -f validation/verify_reference_data.py ]; then
    step "reference data regeneration check"
    pixi run -e ref python validation/verify_reference_data.py
  fi
fi

if [ "${CHECK_CUEQ:-0}" = "1" ] && [ "${CHECK_TORCH:-0}" != "1" ]; then
  # MILESTONE_10 test 6 (and test 7's cueq part with CHECK_LONG=1); with CHECK_TORCH=1 the torch
  # steps above already ran them with CHECK_CUEQ and CUEQ_PYTHON set.
  step "cueq MACE client (pixi ml env, CUEQ_PYTHON=$CUEQ_PYTHON)"
  pixi run -e ml env LIBTORCH_USE_PYTORCH=1 \
    cargo test -p pimd-cli --features torch --test mace_client_cueq
  if [ "${CHECK_LONG:-0}" = "1" ]; then
    pixi run -e ml env LIBTORCH_USE_PYTORCH=1 RAYON_NUM_THREADS=4 \
      cargo test -p pimd-cli --release --features torch,long-validation --test long_mace_client
  fi
fi

printf '\nALL CHECKS PASSED\n'
