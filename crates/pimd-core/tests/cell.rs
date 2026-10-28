//! Tests for `pimd_core::Cell` (SPEC C-1, C-2, C-3).

mod common;

use common::SplitMix64;
use pimd_core::{Cell, CoreError};

const L: [f64; 3] = [10.0, 8.0, 6.0];

fn cell() -> Cell {
    Cell::new(L).unwrap()
}

/// Asserts C-2 for one input: each component of `min_image(d)` lies in [−L/2, L/2) and
/// (d − result)/L is an integer to within 1e-12.
fn check_min_image(c: &Cell, d: [f64; 3]) {
    let r = c.min_image(d);
    let l = c.lengths();
    for k in 0..3 {
        assert!(
            r[k] >= -0.5 * l[k] && r[k] < 0.5 * l[k],
            "min_image({d:?})[{k}] = {} not in [-{h}, {h}) (L = {})",
            r[k],
            l[k],
            h = 0.5 * l[k]
        );
        let n = (d[k] - r[k]) / l[k];
        assert!(
            (n - n.round()).abs() <= 1e-12,
            "min_image({d:?})[{k}] = {}: (d - r)/L = {n} is not an integer",
            r[k]
        );
    }
}

/// Asserts C-3 for one input: each component of `wrap(r)` lies in [0, L) and
/// (r − result)/L is an integer to within 1e-12.
fn check_wrap(c: &Cell, r: [f64; 3]) {
    let w = c.wrap(r);
    let l = c.lengths();
    for k in 0..3 {
        assert!(
            w[k] >= 0.0 && w[k] < l[k],
            "wrap({r:?})[{k}] = {} not in [0, {})",
            w[k],
            l[k]
        );
        let n = (r[k] - w[k]) / l[k];
        assert!(
            (n - n.round()).abs() <= 1e-12,
            "wrap({r:?})[{k}] = {}: (r - w)/L = {n} is not an integer",
            w[k]
        );
    }
}

fn assert_vec_close(got: [f64; 3], want: [f64; 3], what: &str) {
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() <= 1e-12,
            "{what}: got {got:?}, want {want:?}"
        );
    }
}

/// C-1: positive finite lengths are accepted and stored; volume = Lx·Ly·Lz.
#[test]
fn new_accepts_positive_finite_lengths() {
    let c = Cell::new([10.0, 8.0, 6.0]).unwrap();
    assert_eq!(c.lengths(), [10.0, 8.0, 6.0]);
    assert_eq!(c.volume(), 480.0);
    let tiny = Cell::new([1e-3, 2.5, 1e6]).unwrap();
    assert_eq!(tiny.lengths(), [1e-3, 2.5, 1e6]);
}

/// C-1: zero, negative and non-finite lengths in any component are rejected.
#[test]
fn new_rejects_non_positive_or_non_finite_lengths() {
    let bad = [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for &x in &bad {
        for k in 0..3 {
            let mut l = [5.0, 6.0, 7.0];
            l[k] = x;
            let r = Cell::new(l);
            assert!(
                matches!(r, Err(CoreError::InvalidValue { .. })),
                "lengths {l:?}: expected InvalidValue, got {r:?}"
            );
        }
    }
}

/// C-2: hand-picked displacements with L = (10, 8, 6). Expected values are d − nL worked out
/// by hand, with the half-open interval [−L/2, L/2) deciding the boundary cases.
#[test]
fn min_image_hand_picked() {
    let c = cell();
    // Already inside: unchanged.
    assert_vec_close(c.min_image([3.0, -3.0, 2.0]), [3.0, -3.0, 2.0], "inside");
    assert_vec_close(c.min_image([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], "zero");
    // 7 − 10 = −3; −5 + 8 = 3; 4 − 6 = −2.
    assert_vec_close(
        c.min_image([7.0, -5.0, 4.0]),
        [-3.0, 3.0, -2.0],
        "one shift",
    );
    // Large negative: −123.25 + 12·10 = −3.25; 0.5 stays; −17 + 3·6 = 1.
    assert_vec_close(
        c.min_image([-123.25, 0.5, -17.0]),
        [-3.25, 0.5, 1.0],
        "large negative",
    );
    // Multiples of L map to 0.
    assert_vec_close(
        c.min_image([20.0, -16.0, 60.0]),
        [0.0, 0.0, 0.0],
        "multiples",
    );
    assert_vec_close(c.min_image([10.0, 8.0, -6.0]), [0.0, 0.0, 0.0], "one L");
}

/// C-2: exactly ±L/2 (and odd multiples of L/2) map to −L/2, since the interval is [−L/2, L/2).
#[test]
fn min_image_half_box_maps_to_minus_half() {
    let c = cell();
    // +L/2 is outside the half-open interval: 5 − 10 = −5, 4 − 8 = −4, 3 − 6 = −3.
    assert_vec_close(c.min_image([5.0, 4.0, 3.0]), [-5.0, -4.0, -3.0], "+L/2");
    // −L/2 is inside: unchanged.
    assert_vec_close(c.min_image([-5.0, -4.0, -3.0]), [-5.0, -4.0, -3.0], "-L/2");
    // 25 = 2.5 L → −5; −12 = −1.5 L → −4; 9 = 1.5 L → −3.
    assert_vec_close(
        c.min_image([25.0, -12.0, 9.0]),
        [-5.0, -4.0, -3.0],
        "odd L/2",
    );
    // −15 = −1.5 L → −5.
    assert_vec_close(
        c.min_image([-15.0, 12.0, -9.0]),
        [-5.0, -4.0, -3.0],
        "neg odd L/2",
    );
}

/// C-2: the interval holds for the floating-point neighbours of 0, ±L/2 and ±L, where a naive
/// `d − L·floor(d/L + 1/2)` can round onto the wrong side; checked for L = 1 and L = (10, 8, 6).
#[test]
fn min_image_adjacent_floats_at_boundaries() {
    for c in [Cell::new([1.0, 1.0, 1.0]).unwrap(), cell()] {
        let l = c.lengths();
        let mut inputs = Vec::new();
        for base in [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5] {
            let x = [base * l[0], base * l[1], base * l[2]];
            inputs.push(x);
            inputs.push(x.map(f64::next_up));
            inputs.push(x.map(f64::next_down));
        }
        for d in inputs {
            check_min_image(&c, d);
        }
    }
}

/// C-2: property test over 20 000 seeded random vectors (|d| up to 50 L and near the box):
/// result in [−L/2, L/2) and (d − result)/L integral to 1e-12.
#[test]
fn min_image_property_random() {
    let c = Cell::new([10.0, 7.5, 13.3]).unwrap();
    let l = c.lengths();
    let mut rng = SplitMix64::new(0xC2C2_0001);
    for n in 0..20_000 {
        let scale = if n % 2 == 0 { 50.0 } else { 1.0 };
        let d = [0, 1, 2].map(|k| rng.range(-scale * l[k], scale * l[k]));
        check_min_image(&c, d);
    }
}

/// C-3: hand-picked positions with L = (10, 8, 6). Expected values are r − nL by hand.
#[test]
fn wrap_hand_picked() {
    let c = cell();
    assert_vec_close(c.wrap([3.0, 1.0, 5.5]), [3.0, 1.0, 5.5], "inside");
    assert_vec_close(c.wrap([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], "zero");
    // −0.5 + 10 = 9.5; −8 + 8 = 0; 13 − 2·6 = 1.
    assert_vec_close(c.wrap([-0.5, -8.0, 13.0]), [9.5, 0.0, 1.0], "mixed");
    // 25 − 2·10 = 5; −12 + 2·8 = 4; −6.5 + 2·6 = 5.5.
    assert_vec_close(c.wrap([25.0, -12.0, -6.5]), [5.0, 4.0, 5.5], "several L");
    // Multiples of L map to 0 (L itself is outside [0, L)).
    assert_vec_close(c.wrap([10.0, 8.0, 6.0]), [0.0, 0.0, 0.0], "one L");
    assert_vec_close(
        c.wrap([-10.0, -16.0, -60.0]),
        [0.0, 0.0, 0.0],
        "neg multiples",
    );
    // L/2 stays at L/2 (wrap is not min_image); −L/2 → L/2.
    assert_vec_close(c.wrap([5.0, 4.0, 3.0]), [5.0, 4.0, 3.0], "+L/2");
    assert_vec_close(c.wrap([-5.0, -4.0, -3.0]), [5.0, 4.0, 3.0], "-L/2");
}

/// C-3: the interval holds for the floating-point neighbours of 0, ±L/2, ±L and 2L, including
/// tiny negatives such as −1e-17 and the smallest negative subnormal, where r + L rounds to L.
#[test]
fn wrap_adjacent_floats_at_boundaries() {
    for c in [Cell::new([1.0, 1.0, 1.0]).unwrap(), cell()] {
        let l = c.lengths();
        let mut inputs = vec![
            [-1e-17; 3],
            [-f64::MIN_POSITIVE; 3],
            [f64::next_down(0.0); 3],
        ];
        for base in [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0] {
            let x = [base * l[0], base * l[1], base * l[2]];
            inputs.push(x);
            inputs.push(x.map(f64::next_up));
            inputs.push(x.map(f64::next_down));
        }
        for r in inputs {
            check_wrap(&c, r);
        }
    }
}

/// C-3: property test over 20 000 seeded random vectors: result in [0, L) and
/// (r − result)/L integral to 1e-12.
#[test]
fn wrap_property_random() {
    let c = Cell::new([10.0, 7.5, 13.3]).unwrap();
    let l = c.lengths();
    let mut rng = SplitMix64::new(0xC3C3_0001);
    for n in 0..20_000 {
        let scale = if n % 2 == 0 { 50.0 } else { 1.0 };
        let r = [0, 1, 2].map(|k| rng.range(-scale * l[k], scale * l[k]));
        check_wrap(&c, r);
    }
}
