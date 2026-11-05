//! Long validation: PA-CMD centroid correlations in the double well against i-PI's `pa-cmd` mode at
//! two adiabatic frequencies (MILESTONE_7 M7b required test 9; SPEC Y-6, Y-7, Y-2 amended,
//! Y-1 … Y-4, T-1, T-2, T-4, P-2). Feature `long-validation`, run in release by
//! `CHECK_LONG=1 ./scripts/check.sh`.
//!
//! Reference: `data/ipi_doublewell_cmd.json`, written by `data/gen_ipi_doublewell_cmd.py` (i-PI
//! 3.3.1, `normal_modes` mode `pa-cmd`, driving the `doublewell` backend of
//! crates/pimd-socket/tests/clients/ipi_client.py through the socket, under the same PA-CMD
//! segment protocol; versions, seeds and wall time recorded in the file). System and protocol are
//! read from the file: the M7a double well (one atom, m = 1, V = Σ_d (a x_d⁴ − b x_d²), a = 1/8,
//! b = 1, β = 1.5, P = 16), dt = 0.05, PILE-L τ_0 = 1 between segments, Ω = Ω_h = 16^{16/15}/β
//! ≈ 12.83 Eh/ħ (Ω·dt ≈ 0.64) and 2Ω_h, segments under `Trpmd { λ = cmd_lambda = 1 }`; 2 000
//! equilibration steps, cycles of 200 PILE-L steps + redraw + 400-step segment sampled every 4
//! steps (101 lags); i-PI: 1 600 segments per Ω from 8 chains.
//!
//! This side: the same protocol through the library (`m7b_dynamics::run_cycle`, `DoubleWell`),
//! one chain of 6 400 segments per Ω, own seeds. Error bars: Y-4 σ from segments with the
//! 4-segment block check on both sides. Tolerance: 5 σ_combined at every lag. Bias allowance 0
//! (MILESTONE_7 M7b test 9): both codes integrate the same splitting (BAOAB, exact free
//! propagation with the Y-6 masses and frequency Ω, PILE-L/TRPMD friction 2λΩ, centroid NVE in
//! segments) at the same dt and Ω (i-PI's Ω differs by 1e-8 relative, its cm⁻¹ factor being
//! rounded), so the O(dt²) bias is common. Also C_vv(0) = 3/(βm) within 5σ (exact after the
//! amended redraw: the centroid mode is not scaled).
//!
//! Reported, not gated (RESULTS.md): the Ω_h − 2Ω_h difference of C_xx at a few lags, and the
//! exact quantum Kubo-transformed C_xx of the DVR in the locked M7a file `ipi_doublewell.json`.
//!
//! Results with the throwaway reference implementation (release, 43 s): Ω_h worst lag C_xx
//! 1.51 σ_c, C_vv 2.25 σ_c; 2Ω_h C_xx 2.25 σ_c, C_vv 2.28 σ_c (lag 0: engine 2.002 ± 0.015,
//! i-PI 2.081 ± 0.031; exact 3/(βm) = 2); C_xx(0) 9.589 ± 0.010 (Ω_h), 9.564 ± 0.010 (2Ω_h), exact
//! Kubo 9.575; at t = 20: 8.10 ± 0.04 and 8.12 ± 0.04 (difference 0.3 σ_c), exact Kubo 5.72.
#![cfg(feature = "long-validation")]

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;
mod m7b_dynamics;

use lammps::Json;
use m7b_dynamics::{Protocol, Result as CycleResult, Stats, run_cycle};
use pimd_integrators::Thermostat;
use pimd_potentials::DoubleWell;

const IPI_JSON: &str = include_str!("data/ipi_doublewell_cmd.json");
const M7A_JSON: &str = include_str!("data/ipi_doublewell.json");
const N_SEGMENTS: usize = 6400;

fn protocol(j: &Json, omega: f64, seed: u64) -> Protocol {
    let s = j.get("system");
    let p = j.get("protocol");
    let x0 = s.get("x0").nums();
    Protocol {
        mass: s.get("mass").num(),
        beta: s.get("beta").num(),
        n_beads: s.get("n_beads").num() as usize,
        dt: s.get("dt").num(),
        tau0: s.get("tau0").num(),
        omega,
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

/// Y-6, Y-7, Y-2 (amended), Y-1 … Y-4, T-1, T-2, T-4, P-2: PA-CMD C_xx(τ) and C_vv(τ) of one atom
/// in the double well (P = 16, barrier 3 k_B T, λ = 1) agree with i-PI `pa-cmd` under the same
/// protocol within 5 σ_combined at every lag, at Ω_h and at 2Ω_h; C_vv(0) = 3/(βm) within 5σ.
#[test]
fn pacmd_double_well_matches_ipi() {
    let j = Json::parse(IPI_JSON);
    let s = j.get("system");
    let (a, b) = (s.get("a").num(), s.get("b").num());
    let lambda = s.get("cmd_lambda").num();
    let mut runs: Vec<CycleResult> = Vec::new();
    for (key, seed) in [("omega_h", 4001), ("omega_2h", 4002)] {
        let ipi = j.get(key);
        let omega = ipi.get("omega").num();
        let pr = protocol(&j, omega, seed);
        let r = run_cycle(
            &pr,
            Box::new(DoubleWell::new(a, b).unwrap()),
            Thermostat::Trpmd { lambda },
        );
        compare(
            &format!("{key} (Ω = {omega:.4}) C_xx"),
            &r.c_xx,
            ipi.get("c_xx"),
        );
        compare(
            &format!("{key} (Ω = {omega:.4}) C_vv"),
            &r.c_vv,
            ipi.get("c_vv"),
        );
        let want = 3.0 / (pr.beta * pr.mass);
        assert!(
            (r.c_vv.mean[0] - want).abs() <= 5.0 * r.c_vv.sigma[0],
            "{key} C_vv(0) = {} vs 3/(βm) = {want}",
            r.c_vv.mean[0]
        );
        runs.push(r);
    }
    // Report only: Ω_h vs 2Ω_h, and the exact quantum Kubo-transformed C_xx (M7a DVR).
    let lag_t = j.get("lag_time").nums();
    let kubo = Json::parse(M7A_JSON).get("exact_kubo_c_xx").nums();
    for k in [0, 25, 50, 75, 100] {
        let (h, h2) = (&runs[0].c_xx, &runs[1].c_xx);
        eprintln!(
            "t = {:5.1}: C_xx Ω_h {:.4} ± {:.4}, 2Ω_h {:.4} ± {:.4} (difference {:.1} σ_c), exact Kubo {:.4}",
            lag_t[k],
            h.mean[k],
            h.sigma[k],
            h2.mean[k],
            h2.sigma[k],
            (h.mean[k] - h2.mean[k]) / (h.sigma[k].powi(2) + h2.sigma[k].powi(2)).sqrt(),
            kubo[k]
        );
    }
}
