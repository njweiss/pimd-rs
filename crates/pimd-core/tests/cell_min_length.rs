//! Tests for `Cell::min_length` (SPEC C-1, C-4; MILESTONE_4 "Cell::min_length()"): the shortest
//! edge in bohr, bit-identical to the fold it replaces.

use pimd_core::Cell;

/// C-1, C-4: hand values: the shortest of the three edges, whichever position it is in.
#[test]
fn shortest_edge_hand_values() {
    for (l, want) in [
        ([3.0, 1.0, 2.0], 1.0),
        ([1.0, 2.0, 3.0], 1.0),
        ([2.0, 3.0, 0.5], 0.5),
        ([7.25, 7.25, 7.25], 7.25),
        ([1e-3, 1e6, 5.0], 1e-3),
    ] {
        let c = Cell::new(l).unwrap();
        assert_eq!(c.min_length(), want, "lengths {l:?}");
    }
}

/// C-4: bit-identical to `lengths().into_iter().fold(f64::INFINITY, f64::min)`, the expression
/// used by M3's `NeighborList::build`, `Rdf::new` and `LennardJones::evaluate` (MILESTONE_4).
#[test]
fn equals_m3_fold_bitwise() {
    let mut x = 0.123_456_789_f64;
    for _ in 0..200 {
        let mut l = [0.0; 3];
        for v in &mut l {
            // Deterministic pseudo-random lengths in (0.5, 20.5).
            x = (x * 9301.0 + 0.4927).fract();
            *v = 0.5 + 20.0 * x;
        }
        let c = Cell::new(l).unwrap();
        let fold = l.into_iter().fold(f64::INFINITY, f64::min);
        assert_eq!(c.min_length().to_bits(), fold.to_bits(), "lengths {l:?}");
    }
}
