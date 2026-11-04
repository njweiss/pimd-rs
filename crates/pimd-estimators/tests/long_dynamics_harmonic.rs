//! Long validation: RPMD and TRPMD centroid correlations of a harmonic oscillator against the
//! exact finite-dt result (MILESTONE_7 required test 10; SPEC Y-1 … Y-4, T-3, T-4, I-1, I-2).
//! Feature `long-validation`, run in release by `CHECK_LONG=1 ./scripts/check.sh`.
//!
//! System: one atom, m = 1, `Harmonic` with ω = 1, β = 8 (βω = 8), P = 8, dt = 0.1 (ω dt = 0.1),
//! PILE-L τ_0 = 1 between segments. Protocol (`m7_dynamics::run_cycle`): 2 000 equilibration
//! steps, then 8 000 cycles of 100 PILE-L steps, the Y-2 redraw and a 200-step segment sampled
//! every 2 steps (101 lags, τ up to 20). Seeds fixed per run.
//!
//! Exact expectation (hand derivation). For `Harmonic` the centroid decouples from the internal
//! modes (the centroid force is −mω² q̄, tests/rpmd_dynamics.rs in pimd-integrators), and with
//! γ_0 = 0 under `Off` and `Trpmd` it follows the velocity-Verlet map per step,
//!     M = [[1 − h²/2, dt/m], [−mω² dt (1 − h²/4), 1 − h²/2]], h = ω dt,
//! so q̄(n) = M^n_qq q̄(0) + M^n_qp p̄(0) and v̄(n) = M^n_pq q̄(0)/m + M^n_pp v̄(0). At the segment
//! start p̄ is freshly drawn (Y-2) and independent of q̄, with ⟨|v̄|²⟩ = 3/(βm) exactly
//! (v̄ = (1/P) Σ_j p_j/m, Var p_j = mP/β per component). The centroid sampled by BAOAB with the
//! exact free A step is a 1D BAOAB harmonic oscillator, whose configurational marginal is exact
//! for every stable dt (Leimkuhler & Matthews, Appl. Math. Res. Express 2013, 34; J. Chem. Phys.
//! 138, 174102 (2013)): ⟨|q̄|²⟩ = 3/(βmω²), the finite-P (= P-independent, harmonic centroid)
//! value. Hence, at lag τ (n = 2τ steps), with no time-step bias at all:
//!     C_xx(τ) = 3/(βmω²) · [M^n]_qq,   C_vv(τ) = 3/(βm) · [M^n]_pp.
//! Tolerance: 5σ at every lag (σ = max(σ_1, σ_4), m7_dynamics), no bias allowance (none exists).
//!
//! Energy error (MILESTONE_7 test 10, second part): for RPMD the mean |H_P(end) − H_P(start)| per
//! segment (segments.csv definition) is the integrator's O(dt²) shadow-energy error (T-3); at
//! dt/2 = 0.05 (segment and decorrelation steps doubled, same times) it falls by a factor in
//! [3.5, 4.5] (4 × (1 + O(h²)), h ≤ 0.1).
//!
//! Results with the throwaway reference implementation (release, 12 s): RPMD C_xx worst lag 0.68σ,
//! C_vv 0.66σ; TRPMD C_xx 1.33σ, C_vv 0.99σ (σ at lag 0 = 2.5e-3 of 0.375); RPMD mean |ΔH|
//! 7.12e-3 Eh (dt) vs 1.74e-3 Eh (dt/2), ratio 4.09.
#![cfg(feature = "long-validation")]

mod m7_dynamics;

use m7_dynamics::{Protocol, Stats, run_cycle};
use pimd_integrators::Thermostat;
use pimd_potentials::Harmonic;

const MASS: f64 = 1.0;
const OMEGA: f64 = 1.0;
const BETA: f64 = 8.0;

fn protocol(dt: f64, scale: usize, seed: u64, n_segments: usize) -> Protocol {
    Protocol {
        mass: MASS,
        beta: BETA,
        n_beads: 8,
        dt,
        tau0: 1.0,
        n_equil: 2000 * scale,
        decorr: 100 * scale,
        seg: 200 * scale,
        stride: 2 * scale,
        n_lags: 101,
        n_segments,
        seed,
        x0: [0.1, -0.2, 0.05],
    }
}

/// [M^n]_qq and [M^n]_pp of the velocity-Verlet map for ω, m, dt (hand derivation above).
fn vv_power(n: usize, dt: f64) -> (f64, f64) {
    let h2 = (OMEGA * dt).powi(2);
    let m = [
        [1.0 - h2 / 2.0, dt / MASS],
        [
            -MASS * OMEGA * OMEGA * dt * (1.0 - h2 / 4.0),
            1.0 - h2 / 2.0,
        ],
    ];
    let mut a = [[1.0, 0.0], [0.0, 1.0]];
    for _ in 0..n {
        a = [
            [
                m[0][0] * a[0][0] + m[0][1] * a[1][0],
                m[0][0] * a[0][1] + m[0][1] * a[1][1],
            ],
            [
                m[1][0] * a[0][0] + m[1][1] * a[1][0],
                m[1][0] * a[0][1] + m[1][1] * a[1][1],
            ],
        ];
    }
    (a[0][0], a[1][1])
}

fn check(name: &str, s: &Stats, want: &[f64]) {
    let mut worst = (0.0f64, 0usize);
    for (k, (&w, (&m, &sig))) in want.iter().zip(s.mean.iter().zip(&s.sigma)).enumerate() {
        let z = (m - w).abs() / sig;
        if z > worst.0 {
            worst = (z, k);
        }
        assert!(
            (m - w).abs() <= 5.0 * sig,
            "{name} lag {k}: {m:e} vs exact {w:e} (σ {sig:e}, {z:.2} σ)"
        );
    }
    eprintln!(
        "{name}: worst lag {} at {:.2} σ; lag 0 {:.6e} ± {:.1e} (exact {:.6e})",
        worst.1, worst.0, s.mean[0], s.sigma[0], want[0]
    );
}

fn exact(dt: f64, stride: usize) -> (Vec<f64>, Vec<f64>) {
    (0..101)
        .map(|k| {
            let (qq, pp) = vv_power(k * stride, dt);
            (
                3.0 / (BETA * MASS * OMEGA * OMEGA) * qq,
                3.0 / (BETA * MASS) * pp,
            )
        })
        .unzip()
}

/// Y-1 … Y-4, T-3 (RPMD) and T-4 (TRPMD λ = 0.5): C_xx(τ) and C_vv(τ) of one harmonic atom
/// (βω = 8, P = 8) agree with C_xx(0)·[Mⁿ]_qq and C_vv(0)·[Mⁿ]_pp within 5σ at every lag.
#[test]
fn harmonic_correlations_are_exact() {
    let (xx, vv) = exact(0.1, 2);
    for (name, th, seed) in [
        ("RPMD", Thermostat::Off, 1001),
        ("TRPMD", Thermostat::Trpmd { lambda: 0.5 }, 1002),
    ] {
        let pot = Harmonic::new(vec![MASS], OMEGA).unwrap();
        let r = run_cycle(&protocol(0.1, 1, seed, 8000), Box::new(pot), th);
        check(&format!("{name} C_xx"), &r.c_xx, &xx);
        check(&format!("{name} C_vv"), &r.c_vv, &vv);
    }
}

/// T-3 (MILESTONE_7 test 10): the RPMD segment energy error mean |h_end − h_start| (Eh per atom)
/// falls by a factor in [3.5, 4.5] when dt halves (0.1 → 0.05, same segment duration), the
/// O(dt²) scaling of the BAOAB shadow energy.
#[test]
fn rpmd_energy_error_scales_as_dt_squared() {
    let mean = |dt: f64, scale: usize| {
        let pot = Harmonic::new(vec![MASS], OMEGA).unwrap();
        let r = run_cycle(
            &protocol(dt, scale, 1003, 2000),
            Box::new(pot),
            Thermostat::Off,
        );
        r.energy_error.iter().sum::<f64>() / r.energy_error.len() as f64
    };
    let (e1, e2) = (mean(0.1, 1), mean(0.05, 2));
    let ratio = e1 / e2;
    eprintln!("RPMD mean |ΔH|: dt {e1:.4e}, dt/2 {e2:.4e}, ratio {ratio:.3}");
    assert!((3.5..=4.5).contains(&ratio), "ratio {ratio}");
}
