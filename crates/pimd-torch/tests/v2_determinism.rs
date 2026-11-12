//! Determinism with v2 exports (MILESTONE_9 "M9a: Required tests" 5; SPEC D-3 torch note, P-6).
//!
//! D-3: on CPU, results for a fixed libtorch thread count are bitwise reproducible, including the
//! first call after a (re)load, and do not depend on the rayon thread count; the v2 E0 sum runs in
//! f64 in atom order (brief), so it adds nothing that depends on threads, beads or chunks.
//! "Threads 1 and 4 bitwise" is read as in the M6/M8a determinism tests: at libtorch threads 1 and
//! at 4, repeated calls (first and second included), calls under rayon pools of 1 and 4 threads
//! and the first call after a reload are bitwise equal (D-3 does not promise equality between
//! different libtorch thread counts for a real model). For the stub (`stub_v2_ok.pt`, constant
//! interaction energy, so the energy is the E0 sum alone) the results at libtorch threads 1 and 4
//! are also bitwise equal to each other, and every bead's energy is the same.
//!
//! The libtorch thread count is a process-wide setting, so this file holds a single test that runs
//! its cases one after another.
#![cfg(feature = "torch")]

mod common;
mod v2_common;

use common::{bitwise_eq, displaced, eval, max_diff, system};
use pimd_core::Beads;
use pimd_torch::Dtype;
use v2_common::{bige0_v2, bits_eq, load_path, stub, v2_export};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

/// D-3, P-6: v2 exports (the tiny model: 4-water box P = 7, c = 3; the big-E0 model: water P = 16
/// displaced by up to ±0.1 bohr, c = 5), float64 and float32, libtorch threads 1 and 4: 10 calls
/// bitwise equal (rayon 1 and 4 alternating), the first call after a reload equal to them; the stub
/// gives identical energies for every bead, at threads 1 and 4 alike.
#[test]
fn v2_calls_are_bitwise_repeatable() {
    let bx = system("waterbox");
    let box7 = Beads::from_vec(7, bx.n, bx.q.as_slice()[..7 * bx.n * 3].to_vec()).unwrap();
    let wat = system("water");
    let wat16 = displaced(&wat.base, 16, 0.1, 31);
    let cases = [
        (
            "tiny v2, 4-water box P = 7, c = 3",
            v2_export(),
            &bx.z,
            &box7,
            bx.cell,
            3,
        ),
        (
            "big-E0 v2, water P = 16, c = 5",
            bige0_v2(),
            &wat.z,
            &wat16,
            None,
            5,
        ),
    ];
    for threads in [1, 4] {
        for dtype in [Dtype::F64, Dtype::F32] {
            for (name, path, z, q, cell, c) in &cases {
                let what = format!("{name}, {dtype:?}, {threads} libtorch thread(s)");
                let mk = || {
                    load_path(path, z, dtype, threads)
                        .with_beads_per_graph(*c)
                        .unwrap()
                };
                let mut pot = mk();
                let first = pool(1).install(|| eval(&mut pot, q, cell.as_ref()));
                for call in 2..=10 {
                    let rayon_threads = if call % 2 == 0 { 4 } else { 1 };
                    let next = pool(rayon_threads).install(|| eval(&mut pot, q, cell.as_ref()));
                    let (de, df) = max_diff(&next, &first);
                    assert!(
                        bitwise_eq(&next, &first),
                        "{what}: call {call} ({rayon_threads} rayon threads) differs from call 1 \
                         (|dE| {de:e}, |dF| {df:e})"
                    );
                }
                let mut again = mk();
                let fresh = pool(4).install(|| eval(&mut again, q, cell.as_ref()));
                assert!(
                    bitwise_eq(&fresh, &first),
                    "{what}: first call after reloading differs"
                );
            }
        }
    }

    // The stub: the energy is the E0 sum (plus a constant), identical for every bead and for
    // libtorch threads 1 and 4.
    let z = [8u8, 1, 1, 6, 7, 1];
    let q = displaced(
        &[
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [0.0, 1.8, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
        ],
        6,
        0.1,
        7,
    );
    let mut per_threads = Vec::new();
    for threads in [1, 4] {
        let mut pot = load_path(&stub("ok"), &z, Dtype::F64, threads)
            .with_beads_per_graph(4)
            .unwrap();
        let got = pool(threads).install(|| eval(&mut pot, &q, None));
        assert!(
            got.energies
                .iter()
                .all(|e| e.to_bits() == got.energies[0].to_bits()),
            "stub, {threads} thread(s): bead energies differ: {:?}",
            got.energies
        );
        per_threads.push(got.energies);
    }
    assert!(
        bits_eq(&per_threads[0], &per_threads[1]),
        "stub: libtorch threads 1 and 4 differ"
    );
}
