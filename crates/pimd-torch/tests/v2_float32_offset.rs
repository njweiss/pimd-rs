//! float32 offset removed by v2 (MILESTONE_9 "M9a: Required tests" 2; SPEC P-6 v2 sentence, D-1).
//!
//! Model: the tiny model with its E0 table replaced by `big_e0_ev` of `data/tiny_mace_v2.json`
//! (H +1000000.03, C −2000000.05, N −1500000.3, O −2002069.23 eV; `gen_tiny_mace_v2.py`). Mixed
//! signs make every per-bead total small (water and ethanol ≈ −2070 eV, the 4-water box
//! ≈ −8282 eV) while every E0 term and partial sum is ≈ 1e6 eV, where a float32 ulp is 0.0625 eV.
//! The v1 float32 path sums e0 inside the model in float32 (MACE `scatter_sum` in the model dtype),
//! so it misses the float64 energy by many float32 ulp of |E|: the generator measured at least
//! 327 ulp (water, 4-water box) and 614 ulp (ethanol) on these geometries (`bige0_premise`), and
//! this test checks again that v1 lies outside the tolerance below.
//!
//! Expected float64 energy (independent of the code under test): Python MACE's float64 energy of
//! the tiny model (`mace_ref.json`, `energies_ev`) plus Σ_i (E0_big − E0_tiny)(Z_i), summed here
//! in f64 (the generator checked the v1 float64 export against it to 1e-8 eV).
//!
//! Tolerance on the v2 float32 energy (derived, set by the interaction energy):
//! v2 = f64(E_int,f32)·(eV→Eh) + Σ_i E0(Z_i)·(eV→Eh) in f64. The E0 part is exact up to f64
//! rounding of partial sums ≤ 6e6 eV (≤ N·2^-53·6e6 eV ≤ 1e-8 eV for N ≤ 12), so the error is the
//! float32 error of the interaction energy alone. E_int is a sum of N per-atom energies, each the
//! output of a float32 network; with the per-atom magnitudes bounded by those of the tiny model
//! (|E_int,j| ≈ 1.2–4.9 eV here, |e_i| ≲ 1 eV) the float32 error is a modest multiple of
//! eps32 = 2^-24 times the sum of magnitudes. We allow 2^-24·2^8·max(|E_int,j|, N·1 eV) per bead,
//! i.e. 256 eps32 relative to the larger of |E_int,j| and N eV (water 4.6e-5 eV, ethanol 1.4e-4 eV,
//! 4-water box 1.8e-4 eV). Measured with the throwaway reference implementation (2026-11-12, CPU,
//! torch 2.13.0): v2 float32 |ΔE| ≤ 4.4e-7 eV (100× inside); v1 float32 |ΔE| = 0.080 eV (water),
//! 0.150 eV (ethanol), 0.320 eV (box), ≥ 400× outside; reading `atomic_energies()` after the
//! float32 cast moves water by 0.08 eV (Σ of the float32 table rounding), also far outside.
//!
//! Forces: bitwise equal to the v1 float32 forces (both from the gradient of the float32 inter_e).
//! libtorch threads 1.
#![cfg(feature = "torch")]

mod common;
mod v2_common;

use common::{EV_PER_EH, eval, system};
use pimd_torch::Dtype;
use v2_common::{bige0_v1, bige0_v2, bits_eq, load_path, v2_json};

/// eps32 = 2^-24 (unit roundoff of float32), times 2^8.
const TOL_FACTOR: f64 = 256.0 / 16_777_216.0;

/// P-6 (v2 sentence), D-1: water, ethanol and the 4-water box (P = 8) of the big-E0 tiny model,
/// float32: |E_v2 − E_ref| ≤ tol_j per bead (tolerance above), |E_v1 − E_ref| > tol_j for every
/// bead; v2 float32 forces bitwise equal to v1 float32 forces; v2 float64 within 1e-8 eV of the
/// reference (f64 rounding of E0 partial sums ≤ 6e6 eV, as above).
#[test]
fn v2_removes_float32_e0_offset() {
    let j = v2_json();
    let zs: Vec<u8> = j
        .get("atomic_numbers")
        .nums()
        .iter()
        .map(|&z| z as u8)
        .collect();
    let tiny = j.get("tiny_e0_ev").nums();
    let big = j.get("big_e0_ev").nums();
    let idx = |z: u8| {
        zs.iter()
            .position(|&m| m == z)
            .expect("element of the tiny model")
    };
    for name in ["water", "ethanol", "waterbox"] {
        let s = system(name);
        let (mut shift, mut sum_tiny) = (0.0, 0.0);
        for &z in &s.z {
            shift += big[idx(z)] - tiny[idx(z)];
            sum_tiny += tiny[idx(z)];
        }
        let v1 = eval(
            &mut load_path(&bige0_v1(), &s.z, Dtype::F32, 1),
            &s.q,
            s.cell.as_ref(),
        );
        let v2 = eval(
            &mut load_path(&bige0_v2(), &s.z, Dtype::F32, 1),
            &s.q,
            s.cell.as_ref(),
        );
        let v2d = eval(
            &mut load_path(&bige0_v2(), &s.z, Dtype::F64, 1),
            &s.q,
            s.cell.as_ref(),
        );
        assert_eq!(v2.energies.len(), s.p);
        for b in 0..s.p {
            let e_ref = s.energies_ev[b] + shift;
            let e_int = s.energies_ev[b] - sum_tiny;
            let tol = TOL_FACTOR * e_int.abs().max(s.n as f64);
            let d2 = (v2.energies[b] * EV_PER_EH - e_ref).abs();
            let d1 = (v1.energies[b] * EV_PER_EH - e_ref).abs();
            let d2d = (v2d.energies[b] * EV_PER_EH - e_ref).abs();
            eprintln!(
                "{name} bead {b}: E_ref {e_ref:.6} eV, E_int {e_int:.4} eV, tol {tol:.2e}; \
                 |dE| v2 f32 {d2:.2e}, v1 f32 {d1:.2e}, v2 f64 {d2d:.2e} eV"
            );
            assert!(
                d2 <= tol,
                "{name} bead {b}: v2 float32 |dE| {d2:e} eV > {tol:e} eV"
            );
            assert!(
                d1 > tol,
                "{name} bead {b}: v1 float32 |dE| {d1:e} eV within {tol:e} eV (test premise)"
            );
            assert!(d2d <= 1e-8, "{name} bead {b}: v2 float64 |dE| {d2d:e} eV");
        }
        assert!(
            bits_eq(v2.forces.as_slice(), v1.forces.as_slice()),
            "{name}: v2 float32 forces differ from v1 float32 forces"
        );
    }
}
