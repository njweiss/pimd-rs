//! `[dynamics]` runs of the CLI driver (MILESTONE_7 required test 9; SPEC Y-1 … Y-5, D-2, D-3,
//! T-3; "Clarifications": production, checkpoint block, outputs, P = 1).
//!
//! Expectations: (i) the run itself, repeated, at other thread counts and through a restart
//! (determinism, D-3); (ii) `dyn_common::reference_dynamics`, the segment cycle written as a plain
//! loop with hand-computed Y-3/Y-4/Y-5 (independent of `CentroidCorrelation`/`green_kubo`);
//! (iii) the FNV-1a of `checkpoint.bin` written by `pimd-rs` at cf79a4d for configs without
//! `[dynamics]`; (iv) unit conversions with CODATA 2018 values written here (U-2).
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod dyn_common;

use std::path::Path;

use common::{BASE, bytes, observables, parse, pool, read_csv, set, tmp};
use dyn_common::{
    BASE_CKPT_FNV1A_CF79A4D, BOX8, CLUSTER, CLUSTER_CKPT_FNV1A_CF79A4D, CORR_HEADER, DYN_RPMD,
    DYN_TRPMD, SEG_HEADER, fnv1a, mean_sem, reference_dynamics, trailing_buffers, trapezoid_d,
    v2_len,
};
use pimd_cli::{Config, Correlations, RunSummary, run, run_from_checkpoint};
use pimd_estimators::CentroidCorrelation;

/// SPEC U-2 (CODATA 2018), written here for checking the converted columns: 1 bohr =
/// 0.529177210903 Å and 1 fs = 41.341373336561 a.u. of time.
const ANGSTROM_PER_BOHR: f64 = 0.529177210903;
const AU_TIME_FS: f64 = 1.0 / 41.341373336561;

fn with(base: &str, dynamics: &str) -> String {
    format!("{base}{dynamics}")
}

/// Bitwise equality of two `Correlations` (NaN-aware: the σ of a single segment is NaN).
fn same_corr(a: &Option<Correlations>, b: &Option<Correlations>) -> bool {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let xx = |c: &Correlations| c.c_xx.as_ref().map(|(x, s)| (bits(x), bits(s)));
            a.dt_lag.to_bits() == b.dt_lag.to_bits()
                && a.n_segments == b.n_segments
                && bits(&a.c_vv) == bits(&b.c_vv)
                && bits(&a.c_vv_sem) == bits(&b.c_vv_sem)
                && bits(&a.diffusion) == bits(&b.diffusion)
                && xx(a) == xx(b)
        }
        _ => false,
    }
}

fn run_in(cfg: &Config, dir: &Path, threads: usize) -> RunSummary {
    pool(threads).install(|| run(cfg, dir)).unwrap()
}

const FILES: [&str; 5] = [
    "observables.csv",
    "correlations.csv",
    "segments.csv",
    "checkpoint.bin",
    "run.log",
];

/// D-2, D-3 (MILESTONE_7 test 9): a TRPMD and an RPMD run of the 8-atom cluster (P = 4, 3 cycles)
/// write byte-identical outputs (observables.csv, correlations.csv, segments.csv, checkpoint.bin,
/// run.log) and equal summaries when repeated and at 1 and 4 threads.
#[test]
fn runs_are_byte_identical_across_repeats_and_threads() {
    for dynamics in [DYN_TRPMD, DYN_RPMD] {
        let cfg = parse(&with(CLUSTER, dynamics));
        let (a, b, c) = (tmp(), tmp(), tmp());
        let sa = run_in(&cfg, a.path(), 1);
        let sb = run_in(&cfg, b.path(), 1);
        let sc = run_in(&cfg, c.path(), 4);
        assert!(same_corr(&sa.correlations, &sb.correlations));
        assert!(same_corr(&sa.correlations, &sc.correlations));
        assert_eq!(sa.mean_e_vir.to_bits(), sb.mean_e_vir.to_bits());
        assert_eq!(sa.mean_e_vir.to_bits(), sc.mean_e_vir.to_bits());
        for f in FILES {
            let fa = bytes(&a.path().join(f));
            assert_eq!(fa, bytes(&b.path().join(f)), "{f} differs between repeats");
            assert_eq!(fa, bytes(&c.path().join(f)), "{f} differs at 4 threads");
        }
    }
}

/// D-3, MILESTONE_7 "Checkpoint block" / "Outputs": 4 cycles uninterrupted equal 2 cycles →
/// checkpoint → 2 more cycles, bitwise: the final checkpoint.bin and correlations.csv (written
/// after the restart from the restored accumulators) are byte-identical, the summaries are equal,
/// and segments.csv and observables.csv of the two legs concatenate to the uninterrupted files
/// (each leg's file holds only its own rows). TRPMD and RPMD.
#[test]
fn restart_at_a_cycle_end_is_bitwise() {
    for dynamics in [DYN_TRPMD, DYN_RPMD] {
        let full_t = set(&with(CLUSTER, dynamics), "sampling", "n_steps", "72");
        let half_t = set(&full_t, "sampling", "n_steps", "36");
        let (full, leg1, leg2) = (tmp(), tmp(), tmp());
        let s_full = run_in(&parse(&full_t), full.path(), 1);
        run_in(&parse(&half_t), leg1.path(), 1);
        let s_rest = pool(1)
            .install(|| {
                run_from_checkpoint(
                    &parse(&full_t),
                    &leg1.path().join("checkpoint.bin"),
                    leg2.path(),
                )
            })
            .unwrap();
        assert!(same_corr(&s_rest.correlations, &s_full.correlations));
        assert_eq!(s_rest.n_steps, s_full.n_steps);
        for f in ["checkpoint.bin", "correlations.csv"] {
            assert_eq!(
                bytes(&leg2.path().join(f)),
                bytes(&full.path().join(f)),
                "{f}: restart differs"
            );
        }
        for f in ["segments.csv", "observables.csv"] {
            let (h_full, r_full) = read_csv(&full.path().join(f));
            let (h1, mut r1) = read_csv(&leg1.path().join(f));
            let (h2, r2) = read_csv(&leg2.path().join(f));
            assert_eq!((&h1, &h2), (&h_full, &h_full));
            assert!(!r1.is_empty() && !r2.is_empty(), "{f}: a leg wrote no rows");
            r1.extend(r2);
            assert_eq!(r1, r_full, "{f}: legs do not concatenate to the full run");
        }
    }
}

/// Parses a correlations.csv: header and numeric rows (NaN allowed).
fn corr_rows(dir: &Path) -> Vec<Vec<f64>> {
    let (header, rows) = read_csv(&dir.join("correlations.csv"));
    assert_eq!(header, CORR_HEADER);
    rows.iter()
        .map(|r| {
            assert_eq!(r.len(), 8);
            r.iter().map(|v| v.parse::<f64>().unwrap()).collect()
        })
        .collect()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    close_s(a, b, tol, 0.0, what);
}

/// |a − b| ≤ tol·max(|b|, scale): relative, with an absolute floor `scale` for values that cross
/// zero (a correlation at a lag where it changes sign).
fn close_s(a: f64, b: f64, tol: f64, scale: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(scale).max(1e-300),
        "{what}: {a:e} vs {b:e}"
    );
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Y-1 … Y-5, D-2 (MILESTONE_7 "Production with `[dynamics]`"): the run of a gas-phase `[dynamics]`
/// config equals the reference cycle (`dyn_common::reference_dynamics`: one RNG for initial
/// momenta, equilibration, decorrelation thermostat, redraws and segment thermostat in the order
/// they occur; centroid samples at segment steps 0, 3, …, 12; C_seg by Y-4, mean and σ over
/// segments, D by Y-5, all by hand). Checked: `RunSummary::correlations` (dt_lag = corr_stride·dt,
/// n_segments, C_vv and σ, C_xx and σ, D) to 1e-12 relative (σ 1e-9), the correlations.csv
/// columns (lag, t_fs = lag·corr_stride·dt in fs, the au columns, D in Å²/ps =
/// D_au·Å²/bohr²/(ps per ħ/Eh), C_xx), and segments.csv (one row per segment, H_P/N after the
/// redraw and at the segment end, 1e-12). TRPMD, RPMD and the RPMD with decorrelation_steps = 0.
#[test]
fn run_matches_reference_cycle() {
    let no_decorr = set(
        &set(
            &with(
                CLUSTER,
                &DYN_RPMD.replace("decorrelation_steps = 6", "decorrelation_steps = 0"),
            ),
            "sampling",
            "n_steps",
            "36",
        ),
        "output",
        "checkpoint_every",
        "12",
    );
    for toml in [with(CLUSTER, DYN_TRPMD), with(CLUSTER, DYN_RPMD), no_decorr] {
        let cfg = parse(&toml);
        let dy = cfg.dynamics.clone().unwrap();
        let dir = tmp();
        let summary = run_in(&cfg, dir.path(), 1);
        let want = reference_dynamics(&cfg);
        let n_seg = want.cvv_seg.len();
        let c = summary
            .correlations
            .expect("correlations of a [dynamics] run");
        assert_eq!(c.n_segments, n_seg);
        close(
            c.dt_lag,
            dy.corr_stride as f64 * cfg.sampling.dt,
            1e-15,
            "dt_lag",
        );
        let (cvv, cvv_sem) = mean_sem(&want.cvv_seg);
        let (cxx, cxx_sem) = mean_sem(&want.cxx_seg);
        let d = trapezoid_d(&cvv, c.dt_lag);
        let (got_xx, got_xx_sem) = c.c_xx.clone().expect("gas phase: C_xx reported");
        let (svv, sxx, svs, sxs, sd) = (
            max_abs(&cvv),
            max_abs(&cxx),
            max_abs(&cvv_sem),
            max_abs(&cxx_sem),
            max_abs(&d),
        );
        for k in 0..dy.n_lags {
            close_s(c.c_vv[k], cvv[k], 1e-12, svv, &format!("C_vv[{k}]"));
            close_s(
                c.c_vv_sem[k],
                cvv_sem[k],
                1e-9,
                svs,
                &format!("C_vv sem[{k}]"),
            );
            close_s(got_xx[k], cxx[k], 1e-12, sxx, &format!("C_xx[{k}]"));
            close_s(
                got_xx_sem[k],
                cxx_sem[k],
                1e-9,
                sxs,
                &format!("C_xx sem[{k}]"),
            );
            close_s(c.diffusion[k], d[k], 1e-12, sd, &format!("D[{k}]"));
        }
        assert_eq!(c.diffusion[0], 0.0);
        assert_eq!(
            (c.c_vv.len(), c.c_vv_sem.len(), c.diffusion.len()),
            (dy.n_lags, dy.n_lags, dy.n_lags)
        );

        let rows = corr_rows(dir.path());
        assert_eq!(rows.len(), dy.n_lags);
        for (k, r) in rows.iter().enumerate() {
            assert_eq!(r[0], k as f64, "lag column");
            let t_fs = (k * dy.corr_stride) as f64 * cfg.sampling.dt * AU_TIME_FS;
            close(r[1], t_fs, 1e-12, "t_fs");
            close_s(r[2], cvv[k], 1e-12, svv, "c_vv_au");
            close_s(r[3], cvv_sem[k], 1e-9, svs, "c_vv_sem_au");
            // bohr²/(ħ/Eh) → Å²/ps: × (Å/bohr)² / (ps per ħ/Eh).
            let conv = ANGSTROM_PER_BOHR * ANGSTROM_PER_BOHR / (AU_TIME_FS * 1e-3);
            close_s(r[4], d[k], 1e-12, sd, "d_au");
            close_s(r[5], d[k] * conv, 1e-12, sd * conv, "d_angstrom2_per_ps");
            if k == 0 {
                assert_eq!((r[4], r[5]), (0.0, 0.0));
            }
            close_s(r[6], cxx[k], 1e-12, sxx, "c_xx_bohr2");
            close_s(r[7], cxx_sem[k], 1e-9, sxs, "c_xx_sem_bohr2");
        }

        let (header, seg) = read_csv(&dir.path().join("segments.csv"));
        assert_eq!(header, SEG_HEADER);
        assert_eq!(seg.len(), n_seg);
        let cyc = dy.decorrelation_steps + dy.segment_steps;
        let mut prev: Option<(i64, i64)> = None;
        for (r, &(h0, h1)) in seg.iter().zip(&want.h) {
            let (sg, st): (i64, i64) = (r[0].parse().unwrap(), r[1].parse().unwrap());
            if let Some((ps, pt)) = prev {
                assert_eq!(sg, ps + 1, "segment numbering");
                assert_eq!(st, pt + cyc as i64, "segment step spacing");
            }
            prev = Some((sg, st));
            close(r[2].parse().unwrap(), h0, 1e-12, "h_start_eh");
            close(r[3].parse().unwrap(), h1, 1e-12, "h_end_eh");
        }
    }
}

/// MILESTONE_7 "Production with `[dynamics]`": observables.csv keeps recording every production
/// step (decorrelation and segment steps alike) at its stride: n_steps/stride rows at steps
/// stride, 2·stride, …, header unchanged.
#[test]
fn observables_cover_every_production_step() {
    let cfg = parse(&with(CLUSTER, DYN_TRPMD));
    let dir = tmp();
    run_in(&cfg, dir.path(), 1);
    let obs = observables(dir.path());
    let s = &cfg.sampling;
    assert_eq!(obs.len(), s.n_steps / s.stride);
    for (k, r) in obs.iter().enumerate() {
        assert_eq!(r[0], ((k + 1) * s.stride) as f64);
    }
}

/// Y-3, Y-4 with a periodic cell (MILESTONE_7 "Outputs", "Checkpoint block"): C_xx is not
/// reported (`c_xx` None, both C_xx columns of correlations.csv `NaN`), but it is accumulated
/// from the stored, unwrapped centroids (Y-3) and saved: checkpoint.bin is the version-2 layout
/// (with its RDF part) followed by two u64-length-prefixed `CentroidCorrelation` buffers, C_vv
/// then C_xx; decoded, they hold the run's segments and equal the reference cycle's C_vv and
/// C_xx (unwrapped q̄; several atoms of BOX8 start outside [0, L)) to 1e-12.
#[test]
fn periodic_run_hides_c_xx_but_saves_it() {
    let cfg = parse(&with(BOX8, DYN_TRPMD));
    let dir = tmp();
    let summary = run_in(&cfg, dir.path(), 1);
    let c = summary.correlations.unwrap();
    assert!(c.c_xx.is_none(), "C_xx reported with a cell");
    for r in corr_rows(dir.path()) {
        assert!(r[6].is_nan() && r[7].is_nan(), "C_xx columns not NaN");
    }
    let raw = bytes(&dir.path().join("checkpoint.bin"));
    let v2 = v2_len(4, 8, Some(10));
    let (bvv, bxx) = trailing_buffers(&raw, v2);
    let (vv, xx) = (
        CentroidCorrelation::from_bytes(&bvv).unwrap(),
        CentroidCorrelation::from_bytes(&bxx).unwrap(),
    );
    let want = reference_dynamics(&cfg);
    assert_eq!(vv.n_segments(), want.cvv_seg.len());
    assert_eq!(xx.n_segments(), want.cxx_seg.len());
    let (mvv, _) = mean_sem(&want.cvv_seg);
    let (mxx, _) = mean_sem(&want.cxx_seg);
    let (svv, sxx) = (max_abs(&mvv), max_abs(&mxx));
    for k in 0..mvv.len() {
        close_s(
            vv.mean()[k],
            mvv[k],
            1e-12,
            svv,
            &format!("saved C_vv[{k}]"),
        );
        close_s(
            xx.mean()[k],
            mxx[k],
            1e-12,
            sxx,
            &format!("saved C_xx[{k}] (unwrapped)"),
        );
        close_s(c.c_vv[k], mvv[k], 1e-12, svv, &format!("C_vv[{k}]"));
    }
}

/// MILESTONE_7 "No change without `[dynamics]`" / "Checkpoint block": without the section a run
/// writes a checkpoint.bin whose FNV-1a equals the one written by `pimd-rs` at cf79a4d for the
/// same config (gas-phase cluster and periodic M4 system), reports no correlations and writes
/// neither correlations.csv nor segments.csv.
#[test]
fn no_section_keeps_m6_checkpoint() {
    for (toml, want) in [
        (CLUSTER, CLUSTER_CKPT_FNV1A_CF79A4D),
        (BASE, BASE_CKPT_FNV1A_CF79A4D),
    ] {
        let dir = tmp();
        let s = run_in(&parse(toml), dir.path(), 1);
        assert!(s.correlations.is_none());
        assert_eq!(fnv1a(&bytes(&dir.path().join("checkpoint.bin"))), want);
        assert!(!dir.path().join("correlations.csv").exists());
        assert!(!dir.path().join("segments.csv").exists());
    }
}

/// MILESTONE_7 "Checkpoint block": with the section (gas phase, no RDF) checkpoint.bin is the
/// version-2 layout of the M6 length followed by the two length-prefixed buffers (C_vv, C_xx),
/// each holding the run's 3 segments.
#[test]
fn section_appends_two_buffers() {
    let dir = tmp();
    run_in(&parse(&with(CLUSTER, DYN_RPMD)), dir.path(), 1);
    let raw = bytes(&dir.path().join("checkpoint.bin"));
    let (bvv, bxx) = trailing_buffers(&raw, v2_len(4, 8, None));
    assert_eq!(
        CentroidCorrelation::from_bytes(&bvv).unwrap().n_segments(),
        3
    );
    assert_eq!(
        CentroidCorrelation::from_bytes(&bxx).unwrap().n_segments(),
        3
    );
}

/// T-4, Y-1 (MILESTONE_7 "P = 1"): at P = 1 TRPMD has no internal modes, so within each segment it
/// follows RPMD exactly; the RNG streams differ (TRPMD still draws in the O step), so only a run
/// with no thermostatted steps after the first segment start compares: with n_equil = 0,
/// decorrelation_steps = 0 and one cycle, the TRPMD and RPMD runs have bitwise equal C_vv, C_xx
/// and segments.csv (the initial draw and the first redraw are the same draws).
#[test]
fn one_bead_trpmd_equals_rpmd_in_a_segment() {
    let one_bead = |dynamics: &str| {
        let t = with(
            CLUSTER,
            &dynamics.replace("decorrelation_steps = 6", "decorrelation_steps = 0"),
        );
        let t = set(&t, "sampling", "n_beads", "1");
        let t = set(&t, "sampling", "n_equil", "0");
        let t = set(&t, "sampling", "n_steps", "12");
        set(&t, "output", "checkpoint_every", "12")
    };
    let (a, b) = (tmp(), tmp());
    let sa = run_in(&parse(&one_bead(DYN_TRPMD)), a.path(), 1);
    let sb = run_in(&parse(&one_bead(DYN_RPMD)), b.path(), 1);
    assert!(sa.correlations.is_some());
    assert!(same_corr(&sa.correlations, &sb.correlations));
    assert_eq!(
        bytes(&a.path().join("segments.csv")),
        bytes(&b.path().join("segments.csv"))
    );
}
