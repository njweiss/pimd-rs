//! Long validation: RPMD and TRPMD centroid correlations in the double well against i-PI
//! (MILESTONE_7 required test 11; SPEC Y-1 … Y-4, T-3, T-4, P-2). Feature `long-validation`,
//! run in release by `CHECK_LONG=1 ./scripts/check.sh`.
//!
//! Reference: `data/ipi_doublewell.json`, written by `data/gen_ipi_doublewell.py` (i-PI 3.3.1
//! driving the `doublewell` backend of crates/pimd-socket/tests/clients/ipi_client.py through the
//! socket, under the same segment protocol; versions and seeds recorded in the file). System and
//! protocol are read from the file: one atom, m = 1, V = Σ_d (a x_d⁴ − b x_d²) with a = 1/8,
//! b = 1 (barrier 2 Eh = 3 k_B T at β = 1.5), P = 16, dt = 0.05, PILE-L τ_0 = 1 between segments,
//! 2 000 equilibration steps, cycles of 200 PILE-L steps + redraw + 400-step segment sampled every
//! 4 steps (101 lags); i-PI: 1 600 segments per method from 8 independent chains.
//!
//! This side: the same protocol through the library (`m7_dynamics::run_cycle`, `DoubleWell`),
//! one chain of 6 400 segments per method, own seeds. Error bars: Y-4 σ from segments with the
//! 4-segment block check on both sides (σ = max(σ_1, σ_4)). Tolerance: 5 σ_combined at every
//! lag, σ_combined = √(σ_engine² + σ_i-PI²). Discretisation bias: both codes integrate the same
//! splitting (BAOAB with the exact free ring-polymer propagator; RPMD segments without O step,
//! TRPMD with γ_k = 2λω_k and γ_0 = 0) at the same dt from the same distribution, so the O(dt²)
//! bias is common and the allowance is 0 (as the M4 neon comparison).
//!
//! The exact quantum Kubo-transformed C_xx(t) of the DVR in the same file is printed for the
//! RESULTS.md comparison only (RPMD is exact only at t = 0 and P → ∞; not a gate).
//!
//! Results with the throwaway reference implementation (release, 51 s): worst lag RPMD C_xx
//! 1.31 σ_c, C_vv 2.22 σ_c; TRPMD C_xx 1.84 σ_c, C_vv 2.65 σ_c; C_xx(0) 9.576 ± 0.011 (RPMD) vs
//! exact Kubo 9.575; at t = 20, RPMD 8.34, TRPMD 7.85, exact Kubo 5.72.
#![cfg(feature = "long-validation")]

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;
mod m7_dynamics;

use lammps::Json;
use m7_dynamics::{Protocol, Stats, run_cycle};
use pimd_integrators::Thermostat;
use pimd_potentials::DoubleWell;

const IPI_JSON: &str = include_str!("data/ipi_doublewell.json");
const N_SEGMENTS: usize = 6400;

fn protocol(j: &Json, seed: u64) -> Protocol {
    let s = j.get("system");
    let p = j.get("protocol");
    let x0 = s.get("x0").nums();
    Protocol {
        mass: s.get("mass").num(),
        beta: s.get("beta").num(),
        n_beads: s.get("n_beads").num() as usize,
        dt: s.get("dt").num(),
        tau0: s.get("tau0").num(),
        n_equil: p.get("n_equil").num() as usize,
        decorr: p.get("decorrelation_steps").num() as usize,
        seg: p.get("segment_steps").num() as usize,
        stride: p.get("corr_stride").num() as usize,
        n_lags: p.get("n_lags").num() as usize,
        n_segments: N_SEGMENTS,
        seed,
        x0: [x0[0], x0[1], x0[2]],
    }
}

fn compare(name: &str, mine: &Stats, ipi: &Json) {
    let (m, s) = (ipi.get("mean").nums(), ipi.get("sigma").nums());
    assert_eq!(m.len(), mine.mean.len(), "{name}: lag count");
    let mut worst = (0.0f64, 0usize);
    for k in 0..m.len() {
        let sc = (mine.sigma[k].powi(2) + s[k].powi(2)).sqrt();
        let z = (mine.mean[k] - m[k]).abs() / sc;
        if z > worst.0 {
            worst = (z, k);
        }
        assert!(
            z <= 5.0,
            "{name} lag {k}: engine {:e} ± {:e} vs i-PI {:e} ± {:e} ({z:.2} σ_c)",
            mine.mean[k],
            mine.sigma[k],
            m[k],
            s[k]
        );
    }
    eprintln!(
        "{name}: worst lag {} at {:.2} σ_c; lag 0 engine {:.5} ± {:.5}, i-PI {:.5} ± {:.5}",
        worst.1, worst.0, mine.mean[0], mine.sigma[0], m[0], s[0]
    );
}

/// Y-1 … Y-4, T-3, T-4, P-2: C_xx(τ) and C_vv(τ) of RPMD and TRPMD (λ = 0.5) for one atom in the
/// double well (P = 16, barrier 3 k_B T) agree with i-PI under the same segment protocol within
/// 5 σ_combined at every lag. Also C_vv(0) = 3/(βm) within 5σ (Y-4: exact after the redraw).
#[test]
fn double_well_matches_ipi() {
    let j = Json::parse(IPI_JSON);
    let s = j.get("system");
    let (a, b) = (s.get("a").num(), s.get("b").num());
    let lambda = s.get("trpmd_lambda").num();
    let kubo = j.get("exact_kubo_c_xx").nums();
    for (name, th, seed) in [
        ("rpmd", Thermostat::Off, 2001),
        ("trpmd", Thermostat::Trpmd { lambda }, 2002),
    ] {
        let pr = protocol(&j, seed);
        let r = run_cycle(&pr, Box::new(DoubleWell::new(a, b).unwrap()), th);
        let ipi = j.get(name);
        compare(&format!("{name} C_xx"), &r.c_xx, ipi.get("c_xx"));
        compare(&format!("{name} C_vv"), &r.c_vv, ipi.get("c_vv"));
        let want = 3.0 / (pr.beta * pr.mass);
        assert!(
            (r.c_vv.mean[0] - want).abs() <= 5.0 * r.c_vv.sigma[0],
            "{name} C_vv(0) = {} vs 3/(βm) = {want}",
            r.c_vv.mean[0]
        );
        // Report only: RPMD/TRPMD C_xx vs the exact quantum Kubo-transformed C_xx (DVR).
        let lag_t = j.get("lag_time").nums();
        for k in [0, 25, 50, 75, 100] {
            eprintln!(
                "{name} t = {:5.1}: C_xx engine {:.4} ± {:.4}, exact Kubo {:.4}",
                lag_t[k], r.c_xx.mean[k], r.c_xx.sigma[k], kubo[k]
            );
        }
    }
}
