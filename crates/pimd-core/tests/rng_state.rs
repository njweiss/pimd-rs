//! Tests for the RNG snapshot (SPEC D-2, D-3; MILESTONE_4 "RNG snapshot"): `Rng::snapshot` /
//! `Rng::restore` continue the exact sequence, `RngState::to_bytes` / `from_bytes` round trip,
//! and malformed bytes are rejected. Expected sequences come from a second generator with the
//! same seed (or from the generator that was snapshotted), never from fixed golden values.

use pimd_core::{Rng, RngState};

/// Draws `n` values from `r`, interleaving normals and uniforms (pattern n % 3 == 0 → uniform),
/// as bit patterns.
fn draws(r: &mut Rng, n: usize) -> Vec<u64> {
    (0..n)
        .map(|k| {
            if k % 3 == 0 {
                r.uniform().to_bits()
            } else {
                r.normal().to_bits()
            }
        })
        .collect()
}

/// D-2, D-3: `restore(snapshot())` continues the same sequence as the original generator, for
/// snapshots taken after 0, 1, 2, 7 and 1001 draws (odd counts included, so a buffered half of a
/// 64-bit word or a partially used block would show), for several seeds.
#[test]
fn restore_continues_same_sequence() {
    for seed in [0u64, 1, 12_345, u64::MAX] {
        for skip in [0usize, 1, 2, 7, 1001] {
            let mut a = Rng::from_seed(seed);
            draws(&mut a, skip);
            let state = a.snapshot();
            let mut b = Rng::restore(&state);
            assert_eq!(
                draws(&mut a, 500),
                draws(&mut b, 500),
                "seed {seed}, snapshot after {skip} draws"
            );
        }
    }
}

/// D-2: `snapshot` does not advance the generator: the stream after a snapshot equals the stream
/// of an identical generator that was never snapshotted.
#[test]
fn snapshot_does_not_advance() {
    let mut a = Rng::from_seed(99);
    let mut b = Rng::from_seed(99);
    draws(&mut a, 13);
    draws(&mut b, 13);
    let _ = a.snapshot();
    let _ = a.snapshot();
    assert_eq!(draws(&mut a, 200), draws(&mut b, 200));
}

/// D-2: a restored generator is independent of the original: drawing from one does not move the
/// other, and two restores of one state give the same stream.
#[test]
fn restored_generators_are_independent() {
    let mut a = Rng::from_seed(5);
    draws(&mut a, 3);
    let s = a.snapshot();
    let mut b = Rng::restore(&s);
    let mut c = Rng::restore(&s);
    let from_b = draws(&mut b, 100);
    let from_a = draws(&mut a, 100);
    let from_c = draws(&mut c, 100);
    assert_eq!(from_a, from_b);
    assert_eq!(from_a, from_c);
}

/// D-2: states differ when the positions differ: different seeds, or the same seed after a
/// different number of draws; equal for the same seed and position (`PartialEq`).
#[test]
fn state_equality_tracks_position() {
    let s0 = Rng::from_seed(1).snapshot();
    let s0b = Rng::from_seed(1).snapshot();
    let s1 = Rng::from_seed(2).snapshot();
    assert_eq!(s0, s0b);
    assert_ne!(s0, s1);
    let mut r = Rng::from_seed(1);
    r.normal();
    assert_ne!(r.snapshot(), s0);
    let mut q = Rng::from_seed(1);
    q.normal();
    assert_eq!(r.snapshot(), q.snapshot());
}

/// D-3: `from_bytes(to_bytes(s)) == s`, and the generator restored from the decoded state
/// continues the original sequence; the encoding is deterministic (same state, same bytes) and has
/// one fixed length for every state.
#[test]
fn bytes_round_trip() {
    let mut lens = Vec::new();
    for seed in [0u64, 7, 12_345, u64::MAX] {
        for skip in [0usize, 1, 64, 1000] {
            let mut a = Rng::from_seed(seed);
            draws(&mut a, skip);
            let s = a.snapshot();
            let bytes = s.to_bytes();
            assert_eq!(bytes, s.clone().to_bytes(), "encoding not deterministic");
            lens.push(bytes.len());
            let back = RngState::from_bytes(&bytes).expect("round trip");
            assert_eq!(back, s, "seed {seed}, skip {skip}");
            let mut b = Rng::restore(&back);
            assert_eq!(draws(&mut a, 300), draws(&mut b, 300));
        }
    }
    assert!(
        lens.iter().all(|&l| l == lens[0] && l > 0),
        "lengths {lens:?}"
    );
}

/// D-3: malformed encodings are rejected with an error, not a panic: empty input, the valid
/// encoding truncated by one byte, extended by one byte, and halved.
#[test]
fn corrupt_bytes_rejected() {
    let mut a = Rng::from_seed(3);
    draws(&mut a, 10);
    let good = a.snapshot().to_bytes();
    assert!(RngState::from_bytes(&[]).is_err(), "empty input accepted");
    assert!(
        RngState::from_bytes(&good[..good.len() - 1]).is_err(),
        "truncated input accepted"
    );
    let mut longer = good.clone();
    longer.push(0);
    assert!(
        RngState::from_bytes(&longer).is_err(),
        "over-long input accepted"
    );
    assert!(
        RngState::from_bytes(&good[..good.len() / 2]).is_err(),
        "half input accepted"
    );
}
