//! D-3: `LennardJones` results are bitwise identical across thread counts.
//!
//! Each evaluation runs inside an explicit rayon pool (`ThreadPoolBuilder::num_threads`, which
//! overrides `RAYON_NUM_THREADS`), with 1 and with 4 threads. Energies, forces and virial must be
//! equal bit for bit (`to_bits`); any float reduction whose order depends on scheduling (e.g. a
//! rayon `sum` over atoms, whose split tree depends on the pool size) breaks this.

mod common;

use common::lammps::Snapshot;
use common::{SplitMix64, beads, random_config, unwrap_randomly};
use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, LennardJones, LjParams, Potential};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

fn assert_bitwise(a: &Evaluation, b: &Evaluation, what: &str) {
    assert_eq!(a.energies.len(), b.energies.len());
    for (j, (x, y)) in a.energies.iter().zip(&b.energies).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: energy of bead {j}: {x} vs {y}"
        );
    }
    for (k, (x, y)) in a
        .forces
        .as_slice()
        .iter()
        .zip(b.forces.as_slice())
        .enumerate()
    {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: force element {k}: {x} vs {y}"
        );
    }
    match (a.virial, b.virial) {
        (Some(v), Some(w)) => {
            for r in 0..3 {
                for c in 0..3 {
                    assert_eq!(v[r][c].to_bits(), w[r][c].to_bits(), "{what}: W[{r}][{c}]");
                }
            }
        }
        (None, None) => {}
        _ => panic!("{what}: virial presence differs"),
    }
}

/// Evaluates `q` with a fresh `LennardJones` inside a pool of `threads` threads.
fn eval_in(threads: usize, params: LjParams, q: &Beads, cell: &Cell) -> Evaluation {
    pool(threads).install(|| {
        let mut pot = LennardJones::new(params).unwrap();
        pot.evaluate(q, Some(cell), None).unwrap()
    })
}

/// D-3, P-3, P-5: the 256-atom LAMMPS snapshot (shift yes and no), 1 thread vs 4 threads,
/// bitwise equal energies, forces and virial; and a 4-thread evaluation repeated is identical.
#[test]
fn snapshot_bitwise_across_thread_counts() {
    let s = Snapshot::load();
    let cell = s.cell();
    for shift in [true, false] {
        let params = LjParams {
            epsilon: s.epsilon,
            sigma: s.sigma,
            cutoff: s.cutoff,
            skin: s.skin,
            shift,
        };
        let one = eval_in(1, params, &s.q, &cell);
        let four = eval_in(4, params, &s.q, &cell);
        let again = eval_in(4, params, &s.q, &cell);
        assert_bitwise(&one, &four, "1 vs 4 threads");
        assert_bitwise(&four, &again, "4 threads twice");
    }
}

/// D-3, N-3: a random periodic system (N = 300, P = 3, unwrapped coordinates, non-cubic cell),
/// 1, 2 and 4 threads, bitwise equal; also after the lists were rebuilt (a second evaluation on
/// the same object after moving every atom by more than skin/2).
#[test]
fn random_system_bitwise_across_thread_counts() {
    let l = [11.0, 12.5, 10.0];
    let cell = Cell::new(l).unwrap();
    let params = LjParams {
        epsilon: 0.8,
        sigma: 1.0,
        cutoff: 2.5,
        skin: 0.4,
        shift: false,
    };
    let mut rng = SplitMix64(0xD3);
    let conf: Vec<Vec<[f64; 3]>> = (0..3)
        .map(|_| {
            let x = random_config(&mut rng, 300, l, 0.85, true);
            unwrap_randomly(&mut rng, &x, l)
        })
        .collect();
    let q1 = beads(&conf);
    let moved: Vec<Vec<[f64; 3]>> = conf
        .iter()
        .map(|c| {
            c.iter()
                .map(|r| [r[0] + 0.3, r[1] - 0.25, r[2] + 0.1])
                .collect()
        })
        .collect();
    let q2 = beads(&moved);
    let run = |threads: usize| {
        pool(threads).install(|| {
            let mut pot = LennardJones::new(params).unwrap();
            let a = pot.evaluate(&q1, Some(&cell), None).unwrap();
            let b = pot.evaluate(&q2, Some(&cell), None).unwrap();
            (a, b)
        })
    };
    let (a1, b1) = run(1);
    for t in [2, 4] {
        let (a, b) = run(t);
        assert_bitwise(&a1, &a, &format!("first evaluation, 1 vs {t} threads"));
        assert_bitwise(&b1, &b, &format!("after rebuild, 1 vs {t} threads"));
    }
}
