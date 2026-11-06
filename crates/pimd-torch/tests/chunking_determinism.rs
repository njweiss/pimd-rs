//! Determinism of chunked evaluation (MILESTONE_8 "M8a: Required tests" 3; SPEC D-3, P-6).
//!
//! D-3: on CPU, results for a fixed libtorch thread count are bitwise reproducible, including the
//! first call (profiling executor disabled, P-6), and do not depend on the number of rayon threads
//! (edges are built in parallel over beads and joined in bead order). With chunks, every call runs
//! the same sequence of chunk graphs, so the same holds: repeated chunked calls are bitwise equal
//! (first and second call too), at libtorch threads 1 and 4, and under rayon pools of 1 and 4
//! threads.
//!
//! The libtorch thread count is a process-wide setting, so this file holds a single test that runs
//! its cases one after another (as the M6 `determinism.rs`).
#![cfg(feature = "torch")]

mod common;

use common::{bitwise_eq, displaced, eval, load, max_diff, system};
use pimd_core::Beads;
use pimd_torch::{Device, Dtype};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

/// D-3, P-6 (MILESTONE_8 test 3): for libtorch threads 1 and 4, float64 and float32, the 4-water
/// box (P = 7, c = 3: chunks 3+3+1) and water (P = 16, the M5 geometry displaced by up to ±0.1
/// bohr, c = 5: 5+5+5+1): the first and second chunked calls are bitwise equal; 10 calls are
/// bitwise equal; calls under rayon pools of 1 and 4 threads are bitwise equal; the first call of
/// a freshly loaded, freshly chunked module is bitwise equal to them.
#[test]
fn chunked_calls_are_bitwise_repeatable() {
    let bx = system("waterbox");
    let box7 = Beads::from_vec(7, bx.n, bx.q.as_slice()[..7 * bx.n * 3].to_vec()).unwrap();
    let wat = system("water");
    let wat16 = displaced(&wat.base, 16, 0.1, 31);
    let cases = [
        ("4-water box P = 7, c = 3", &bx.z, &box7, bx.cell, 3),
        ("water P = 16, c = 5", &wat.z, &wat16, None, 5),
    ];
    for threads in [1, 4] {
        for dtype in [Dtype::F64, Dtype::F32] {
            for (name, z, q, cell, c) in &cases {
                let what = format!("{name}, {dtype:?}, {threads} libtorch thread(s)");
                let mk = || {
                    load(z, Device::Cpu, dtype, threads)
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
}
