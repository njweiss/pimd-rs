//! libtorch's intra-op thread count at evaluation (MILESTONE_6 "After review", S1; SPEC P-6, D-3
//! torch note).
//!
//! libtorch keeps the intra-op thread count per OS thread: a thread that has already run ATen ops
//! keeps its own count when another thread calls `set_num_threads`. `evaluate` therefore applies
//! `threads` on the calling thread on every call. The setting is process-wide state, so this file
//! holds a single test.
#![cfg(feature = "torch")]

mod common;

use common::{bitwise_eq, eval, load, system};
use pimd_torch::{Device, Dtype};
use tch::{Kind, Tensor};

/// P-6, D-3: after `evaluate` the calling thread's intra-op thread count is the potential's
/// `threads` (2), both on the loading thread after something else changed it and on a second
/// thread that had already run ATen ops with another count; the results are bitwise equal.
#[test]
fn evaluate_applies_threads_on_the_calling_thread() {
    let s = system("water");
    let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 2);
    let first = eval(&mut pot, &s.q, None);

    tch::set_num_threads(3);
    let again = eval(&mut pot, &s.q, None);
    assert_eq!(tch::get_num_threads(), 2, "loading thread after evaluate");
    assert!(bitwise_eq(&first, &again), "same thread count, same bits");

    let other = std::thread::scope(|sc| {
        sc.spawn(|| {
            tch::set_num_threads(3);
            let _ = Tensor::ones([64], (Kind::Double, tch::Device::Cpu)).sum(Kind::Double);
            assert_eq!(tch::get_num_threads(), 3, "test setup: second thread at 3");
            let e = eval(&mut pot, &s.q, None);
            assert_eq!(tch::get_num_threads(), 2, "second thread after evaluate");
            e
        })
        .join()
        .expect("second thread")
    });
    assert!(bitwise_eq(&first, &other), "second thread: same bits");
}
