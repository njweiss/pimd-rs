#!/usr/bin/env bash
# Single pass/fail gate. Exits non-zero on the first failure.
#   ./scripts/check.sh              fast checks (what implementers iterate on)
#   CHECK_LONG=1 ./scripts/check.sh also runs long validation and reference-data checks
#   CHECK_IPI=1 (with CHECK_LONG=1) also regenerates the slow i-PI reference (~34 min)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

TESTS_GLOB=':(glob)crates/*/tests/**'

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

if [ "${CHECK_LONG:-0}" = "1" ]; then
  # Every crate that has long tests defines the feature `long-validation`.
  # Revisit --all-features if other features (e.g. a torch backend) are added.
  step "long validation (release)"
  RAYON_NUM_THREADS=4 cargo test --workspace --release --all-features

  if [ -f validation/verify_reference_data.py ]; then
    step "reference data regeneration check"
    pixi run -e ref python validation/verify_reference_data.py
  fi
fi

printf '\nALL CHECKS PASSED\n'
