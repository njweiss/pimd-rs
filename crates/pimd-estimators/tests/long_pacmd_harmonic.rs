//! Long validation: PA-CMD centroid correlations of a harmonic oscillator against the exact
//! finite-dt result, and the Ω dependence of the segment energy error (MILESTONE_7 M7b required
//! test 8; SPEC Y-6, Y-7, Y-2 amended, Y-3, Y-4, T-3, T-4, I-1, I-2). Feature `long-validation`,
//! run in release by `CHECK_LONG=1 ./scripts/check.sh`.
//!
//! System (the M7a test 10 system, long_dynamics_harmonic.rs): one atom, m = 1, `Harmonic` with
//! ω = 1, β = 8, P = 8 (ω_P = 1), dt = 0.1, PILE-L τ_0 = 1 between segments; PA-CMD with Ω = 4
//! (Ω·dt = 0.4; s_k = (ω_k/4)² = 0.037 … 0.25) and segment thermostat `Trpmd { λ = 1 }` (the M7b
//! test value). Protocol (`m7b_dynamics::run_cycle`): the amended Y-2 initial draw, 2 000
//! equilibration steps, then 8 000 cycles of 100 PILE-L steps, the redraw and a 200-step segment
//! sampled every 2 steps (101 lags). Seeds fixed per run.
//!
//! Exact expectation (hand derivation of long_dynamics_harmonic.rs, unchanged by Y-6): for
//! `Harmonic` the centroid decouples from the internal modes; Y-6 keeps its mass physical
//! (s_0 = 1) and T-4 keeps it unthermostatted (γ_0 = 0), so within a segment it follows the
//! velocity-Verlet map M = [[1 − h²/2, dt/m], [−mω² dt (1 − h²/4), 1 − h²/2]], h = ω dt, whatever
//! the internal modes do. At the segment start p̄ is freshly drawn: the amended Y-2 scales mode
//! k = 0 by √s_0 = 1, so ⟨|v̄|²⟩ = 3/(βm) exactly, independent of q̄; the centroid's configurational
//! marginal under BAOAB with the exact free A step is exact (⟨|q̄|²⟩ = 3/(βmω²)) and does not
//! involve the internal masses. Hence C_xx(τ) = 3/(βmω²)·[Mⁿ]_qq and C_vv(τ) = 3/(βm)·[Mⁿ]_pp
//! (n = 2τ), with no time-step bias. Tolerance 5σ at every lag, no bias allowance (none exists).
//!
//! Energy error (second part): under `Off` with Ω the segment energy error |ΔH̃_P| is the BAOAB
//! shadow-energy error of the internal modes, whose effective external stiffness is
//! mω²/(m s_k) = ω²Ω²/ω_k², growing with Ω; at fixed dt the mean |ΔH̃_P| over 2 000 segments is
//! strictly increasing over Ω = 2, 4, 8 (Ω·dt = 0.2, 0.4, 0.8), each step resolved by more than
//! 5σ_combined (σ = SD/√n over segments, with the 4-segment block check).
//!
//! Results with the throwaway reference implementation (release, 6.4 s): C_xx worst lag 1.01 σ,
//! C_vv 1.40 σ (σ at lag 0 = 2.4e-3 of 0.375); mean |ΔH̃_P| = 1.418e-2 ± 2.5e-4 (Ω = 2),
//! 9.998e-2 ± 1.9e-3 (Ω = 4), 5.884e-1 ± 1.1e-2 Eh (Ω = 8): increases of 44.7 and 43.6 σ_c.
#![cfg(feature = "long-validation")]

mod m7b_dynamics;

use m7b_dynamics::{Protocol, Stats, mean_sigma, run_cycle};
use pimd_integrators::Thermostat;
use pimd_potentials::Harmonic;

const MASS: f64 = 1.0;
const OMEGA: f64 = 1.0;
const BETA: f64 = 8.0;
const DT: f64 = 0.1;

fn protocol(omega: f64, seed: u64, n_segments: usize) -> Protocol {
    Protocol {
        mass: MASS,
        beta: BETA,
        n_beads: 8,
        dt: DT,
        tau0: 1.0,
        omega,
        n_equil: 2000,
        decorr: 100,
        seg: 200,
        stride: 2,
        n_lags: 101,
        n_segments,
        seed,
        x0: [0.1, -0.2, 0.05],
    }
}

/// [M^n]_qq and [M^n]_pp of the velocity-Verlet map for ω, m, dt (hand derivation above).
fn vv_power(n: usize) -> (f64, f64) {
    let h2 = (OMEGA * DT).powi(2);
    let m = [
        [1.0 - h2 / 2.0, DT / MASS],
        [
            -MASS * OMEGA * OMEGA * DT * (1.0 - h2 / 4.0),
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

/// Y-6, Y-7, Y-2 (amended), Y-3, Y-4, T-4 with Y-6: PA-CMD (Ω = 4, λ = 1) C_xx(τ) and C_vv(τ) of
/// one harmonic atom (βω = 8, P = 8) agree with C_xx(0)·[Mⁿ]_qq and C_vv(0)·[Mⁿ]_pp within 5σ at
/// every lag.
#[test]
fn pacmd_harmonic_correlations_are_exact() {
    let (xx, vv): (Vec<f64>, Vec<f64>) = (0..101)
        .map(|k| {
            let (qq, pp) = vv_power(2 * k);
            (
                3.0 / (BETA * MASS * OMEGA * OMEGA) * qq,
                3.0 / (BETA * MASS) * pp,
            )
        })
        .unzip();
    let pot = Harmonic::new(vec![MASS], OMEGA).unwrap();
    let r = run_cycle(
        &protocol(4.0, 3001, 8000),
        Box::new(pot),
        Thermostat::Trpmd { lambda: 1.0 },
    );
    check("PA-CMD C_xx", &r.c_xx, &xx);
    check("PA-CMD C_vv", &r.c_vv, &vv);
}

/// Y-6, T-3, I-1, I-2: under `Off` with Ω = 2, 4, 8 at dt = 0.1 the mean segment |ΔH̃_P| is
/// strictly increasing, each step resolved by > 5σ_combined.
#[test]
fn pacmd_energy_error_grows_with_omega() {
    let mut prev: Option<(f64, f64, f64)> = None;
    for (omega, seed) in [(2.0, 3011), (4.0, 3012), (8.0, 3013)] {
        let pot = Harmonic::new(vec![MASS], OMEGA).unwrap();
        let r = run_cycle(&protocol(omega, seed, 2000), Box::new(pot), Thermostat::Off);
        let (m, s) = mean_sigma(&r.energy_error, &format!("|ΔH̃| at Ω = {omega}"));
        eprintln!(
            "Ω = {omega} (Ω·dt = {:.1}): mean |ΔH̃| = {m:.4e} ± {s:.1e} Eh",
            omega * DT
        );
        if let Some((w0, m0, s0)) = prev {
            let z = (m - m0) / (s * s + s0 * s0).sqrt();
            eprintln!("  Ω {w0} → {omega}: increase {z:.1} σ_c");
            assert!(
                z > 5.0,
                "Ω {w0} → {omega}: |ΔH̃| {m0:e} → {m:e} ({z:.2} σ_c)"
            );
        }
        prev = Some((omega, m, s));
    }
}
