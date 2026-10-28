//! Tests for `pimd_core::units` (SPEC U-2, U-4).

use pimd_core::CoreError;
use pimd_core::units::{self, beta_from_kelvin};

/// U-2: every constant equals its CODATA 2018 value as written in SPEC §1, exactly.
#[test]
fn constants_match_spec_values() {
    // Literals copied from SPEC U-2.
    assert_eq!(units::K_B_HARTREE_PER_KELVIN, 3.1668115634556e-6);
    assert_eq!(units::AMU_IN_ELECTRON_MASSES, 1822.888486209);
    assert_eq!(units::BOHR_IN_ANGSTROM, 0.529177210903);
    assert_eq!(units::FS_IN_ATOMIC_TIME, 41.341373336561);
    assert_eq!(units::HARTREE_IN_WAVENUMBERS, 219474.6313632);
    assert_eq!(units::HARTREE_IN_KCAL_PER_MOL, 627.5094740631);
}

/// U-4: β(300 K) = 1/(k_B · 300) with k_B from SPEC U-2, to relative 1e-14.
#[test]
fn beta_at_300_kelvin() {
    // Expected value: the U-4 formula evaluated with the U-2 literal (not the crate constant).
    let expected = 1.0 / (3.1668115634556e-6 * 300.0);
    let beta = beta_from_kelvin(300.0).unwrap();
    assert!(
        ((beta - expected) / expected).abs() <= 1e-14,
        "beta = {beta:e}, expected {expected:e}"
    );
}

/// U-4: β·k_B·T = 1 over several decades of temperature, to relative 1e-14.
#[test]
fn beta_times_kt_is_one() {
    for t in [1e-3, 0.5, 1.0, 4.2, 77.0, 1000.0, 5.0e4] {
        let beta = beta_from_kelvin(t).unwrap();
        let x = beta * 3.1668115634556e-6 * t;
        assert!((x - 1.0).abs() <= 1e-14, "T = {t}: beta*kB*T = {x}");
    }
}

/// U-4: zero, negative and non-finite temperatures are errors (`CoreError::InvalidValue`).
#[test]
fn beta_rejects_non_positive_or_non_finite() {
    for t in [
        0.0,
        -0.0,
        -1.0,
        -300.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let r = beta_from_kelvin(t);
        assert!(
            matches!(r, Err(CoreError::InvalidValue { .. })),
            "T = {t}: expected InvalidValue, got {r:?}"
        );
    }
}
