//! `MacePotential` against Python MACE (MILESTONE_6 "Per evaluation vs Python MACE", "Units";
//! SPEC P-6, U-3, D-1, S-3, S-4, P-5).
//!
//! Expected values: `data/mace_ref.json` from `data/gen_mace_ref.py` (MACECalculator on the
//! committed tiny model, mace's own neighbour list, float64 CPU; mace-torch 0.3.17, torch 2.13.0,
//! versions recorded in the file). Systems: water, ethanol and the 4-water box (unwrapped
//! positions, 224 of 416 edges cross the boundary), 8 displaced beads each, evaluated as one batch.
//!
//! Tolerances (brief): float64 CPU |ΔE| ≤ 1e-12·|E| per bead and max|ΔF| ≤ 1e-10·max|F| per bead
//! (spike measurement: ΔE = 0, ΔF ≤ 2e-15·max|F|); float32 against the float64 reference
//! |ΔE| ≤ 5e-3 eV (converted to Eh) and max|ΔF| ≤ 1e-4·max|F| (spike: 1.5e-3 eV, 4e-6).
//! libtorch intra-op threads 1 in every test of this file.
#![cfg(feature = "torch")]

mod common;

use common::{BOHR_A, EV_PER_EH, SYSTEMS, errors, eval, load, max_abs, system};
use pimd_core::Beads;
use pimd_torch::{Device, Dtype};

/// P-6, U-3, S-3, S-4, P-5: float64 on CPU reproduces Python MACE for water, ethanol and the
/// periodic 4-water box (8 beads each, one batched call): |ΔE| ≤ 1e-12·|E| and
/// max|ΔF| ≤ 1e-10·max|F| per bead; one energy per bead; `virial` is `None`.
#[test]
fn float64_cpu_matches_python_mace() {
    for name in SYSTEMS {
        let s = system(name);
        let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
        let got = eval(&mut pot, &s.q, s.cell.as_ref());
        assert!(got.virial.is_none(), "{name}: P-6/P-5: virial is None");
        let (er, ea, fr) = errors(&got, &s.energies, &s.forces);
        eprintln!("{name} f64 cpu: |dE|/|E| {er:.2e} ({ea:.2e} Eh), |dF|/max|F| {fr:.2e}");
        assert!(er <= 1e-12, "{name}: energy relative error {er:e}");
        assert!(fr <= 1e-10, "{name}: force relative error {fr:e}");
    }
}

/// P-6, D-1: float32 inside the model, f64 in and out: against the float64 Python reference
/// |ΔE| ≤ 5e-3 eV and max|ΔF| ≤ 1e-4·max|F| per bead (brief tolerance; spike measured 1.5e-3 eV
/// and 4e-6 for MACE-OFF23, 8e-4 eV and 4e-6 for this tiny model).
#[test]
fn float32_cpu_within_float32_tolerance() {
    for name in SYSTEMS {
        let s = system(name);
        let mut pot = load(&s.z, Device::Cpu, Dtype::F32, 1);
        let got = eval(&mut pot, &s.q, s.cell.as_ref());
        let (_, ea, fr) = errors(&got, &s.energies, &s.forces);
        eprintln!(
            "{name} f32 cpu: |dE| {:.2e} eV, |dF|/max|F| {fr:.2e}",
            ea * EV_PER_EH
        );
        assert!(
            ea <= 5e-3 / EV_PER_EH,
            "{name}: energy error {} eV",
            ea * EV_PER_EH
        );
        assert!(fr <= 1e-4, "{name}: force relative error {fr:e}");
    }
}

/// D-1, P-6: with `Dtype::F32` the model itself runs in float32 (the cast happens on the way in),
/// not only its outputs, and not a float64 model fed with float32-rounded positions:
/// (a) forces: a float32 rounding of a float64 result changes no force component by more than
///     2⁻²⁴ relative (plus ~1e-16 from the unit conversion), whereas the float32 model differs
///     from the float64 one by float32 rounding accumulated through the network. Criterion: some
///     force component (three systems, 24 beads) differs from the float64 result by more than
///     4·2⁻²⁴ of its own magnitude (reference implementation: 5e-4; outputs-only cast: 5.3e-8).
/// (b) energies: a float64 model (on float32-rounded or exact positions) whose energy is rounded
///     to float32 gives exactly the float32 rounding of the float64 energy in eV (input rounding
///     moves E by ≪ 1 float32 ulp of |E| ≈ 2 000–8 000 eV), whereas the float32 model's energy, a
///     float32 sum through the network, is off by about one ulp or more (spike: up to 3 ulp).
///     Criterion: for some bead, E_f32 [eV] as f32 ≠ E_f64 [eV] as f32 (eV = Eh · 27.211386245988).
#[test]
fn float32_runs_the_model_in_float32() {
    let mut worst = 0.0f64;
    let (mut beads, mut off) = (0, 0);
    for name in SYSTEMS {
        let s = system(name);
        let a = eval(
            &mut load(&s.z, Device::Cpu, Dtype::F64, 1),
            &s.q,
            s.cell.as_ref(),
        );
        let b = eval(
            &mut load(&s.z, Device::Cpu, Dtype::F32, 1),
            &s.q,
            s.cell.as_ref(),
        );
        for (x, y) in a.forces.as_slice().iter().zip(b.forces.as_slice()) {
            if *x != 0.0 {
                worst = worst.max((x - y).abs() / x.abs());
            }
        }
        for (x, y) in a.energies.iter().zip(&b.energies) {
            beads += 1;
            if (x * EV_PER_EH) as f32 != (y * EV_PER_EH) as f32 {
                off += 1;
            }
        }
    }
    let ulp = 2f64.powi(-24);
    eprintln!(
        "largest per-component f32/f64 force difference: {worst:.2e} (2^-24 = {ulp:.2e}); \
         {off} of {beads} float32 energies differ from the float32 rounding of the float64 one"
    );
    assert!(
        worst > 4.0 * ulp,
        "D-1: float32 forces look like rounded float64 results ({worst:e}): the model did not run \
         in float32"
    );
    assert!(
        off > 0,
        "D-1: every float32 energy is the float32 rounding of the float64 energy: the model did not \
         run in float32"
    );
}

/// U-3, P-6: units on one bead, by hand. The water molecule at the M5 start geometry, written in
/// Å here (O 0 0 0.1173; H 0 ±0.7572 −0.4692) and converted with x / 0.529177210903; expected
/// energy E [Eh] = E [eV] · (1 / 27.211386245988) and forces F [Eh/bohr] = F [eV/Å] ·
/// 0.529177210903 / 27.211386245988 from the Python values (`water_single`, P = 1), to
/// 1e-12·|E| and 1e-10·max|F|. The cutoff is r_max = 5 Å → 5 / 0.529177210903 bohr (1e-15).
#[test]
fn units_by_hand_on_one_bead() {
    let s = system("water_single");
    let angstrom = [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ];
    let mut q = Beads::zeros(1, 3);
    for (i, x) in angstrom.iter().enumerate() {
        q.set(0, i, x.map(|c| c / 0.529177210903));
    }
    let mut pot = load(&[8, 1, 1], Device::Cpu, Dtype::F64, 1);
    let r = 5.0 / 0.529177210903;
    assert!(
        (pot.cutoff() - r).abs() <= 1e-15 * r,
        "cutoff {} bohr, want {r} (5 Å)",
        pot.cutoff()
    );
    let got = eval(&mut pot, &q, None);
    let e_want = s.energies_ev[0] * (1.0 / 27.211386245988);
    assert!(
        (got.energies[0] - e_want).abs() <= 1e-12 * e_want.abs(),
        "energy {} Eh, hand conversion {e_want} Eh",
        got.energies[0]
    );
    let r = common::reference();
    let f_ev = r
        .at(&["systems", "water_single", "forces_ev_per_angstrom"])
        .nums();
    let f_want: Vec<f64> = f_ev
        .iter()
        .map(|f| f * 0.529177210903 / 27.211386245988)
        .collect();
    let fmax = max_abs(f_want.iter().copied());
    let df = max_abs(
        got.forces
            .as_slice()
            .iter()
            .zip(&f_want)
            .map(|(a, b)| a - b),
    );
    assert!(
        df <= 1e-10 * fmax,
        "forces differ from hand conversion by {df:e} Eh/bohr"
    );
    // The constants of the common module are the same literals.
    assert_eq!(BOHR_A, 0.529177210903);
    assert_eq!(EV_PER_EH, 27.211386245988);
}

/// P-6: `cutoff()` is the model's r_max (5 Å in the reference JSON) in bohr.
#[test]
fn cutoff_is_model_r_max_in_bohr() {
    let r = common::reference();
    let r_ang = r.at(&["model", "r_max_angstrom"]).num();
    assert_eq!(r_ang, 5.0);
    let pot = load(&[1, 6, 7, 8], Device::Cpu, Dtype::F64, 1);
    let want = r_ang / BOHR_A;
    assert!((pot.cutoff() - want).abs() <= 1e-15 * want);
    let pot32 = load(&[1, 6, 7, 8], Device::Cpu, Dtype::F32, 1);
    assert!(
        (pot32.cutoff() - want).abs() <= 1e-15 * want,
        "same cutoff for Dtype::F32"
    );
}
