//! PA-CMD runs of the CLI driver (MILESTONE_7 M7b required test 7; SPEC Y-1, Y-2, Y-6, Y-7, D-2,
//! D-3; M7b "Clarifications": checkpoint layout unchanged, segments.csv with H̃_P).
//!
//! Expectations: (i) the run itself, repeated, at other thread counts and through a restart
//! (determinism, D-3); (ii) `pacmd_common::reference_pacmd`, the PA-CMD cycle as a plain loop with
//! H̃_P by hand and the hand-written Y-3/Y-4/Y-5 of the locked M7a helpers; (iii) FNV-1a of the
//! outputs of M7a runs written by `pimd-rs` at e6008c9.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod dyn_common;
mod pacmd_common;

use std::path::Path;

use common::{bytes, parse, pool, read_csv, set, tmp};
use dyn_common::{
    CLUSTER, CORR_HEADER, DYN_RPMD, DYN_TRPMD, SEG_HEADER, fnv1a, mean_sem, trailing_buffers,
    trapezoid_d, v2_len,
};
use pacmd_common::{
    CLUSTER_RPMD_FNV1A_E6008C9, CLUSTER_TRPMD_FNV1A_E6008C9, DYN_PACMD, reference_pacmd,
};
use pimd_cli::{Config, RunSummary, run, run_from_checkpoint};
use pimd_estimators::CentroidCorrelation;

fn with(dynamics: &str) -> String {
    format!("{CLUSTER}{dynamics}")
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

/// D-2, D-3 (M7b test 7): a PA-CMD run of the 8-atom cluster (P = 4, 3 cycles) writes
/// byte-identical outputs (observables.csv, correlations.csv, segments.csv, checkpoint.bin,
/// run.log) and bitwise equal summaries when repeated and at 1 and 4 threads.
#[test]
fn pa_cmd_run_is_byte_identical_across_repeats_and_threads() {
    let cfg = parse(&with(DYN_PACMD));
    let (a, b, c) = (tmp(), tmp(), tmp());
    let sa = run_in(&cfg, a.path(), 1);
    let sb = run_in(&cfg, b.path(), 1);
    let sc = run_in(&cfg, c.path(), 4);
    let bits = |s: &RunSummary| {
        let c = s.correlations.as_ref().unwrap();
        c.c_vv
            .iter()
            .chain(&c.diffusion)
            .chain(&c.c_xx.as_ref().unwrap().0)
            .map(|x| x.to_bits())
            .chain([s.mean_e_vir.to_bits(), s.mean_e_prim.to_bits()])
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&sa), bits(&sb));
    assert_eq!(bits(&sa), bits(&sc));
    for f in FILES {
        let fa = bytes(&a.path().join(f));
        assert_eq!(fa, bytes(&b.path().join(f)), "{f} differs between repeats");
        assert_eq!(fa, bytes(&c.path().join(f)), "{f} differs at 4 threads");
    }
}

/// D-3, M7b "Clarifications" (checkpoint layout unchanged; Ω comes from the config): 4 cycles
/// uninterrupted equal 2 cycles → checkpoint → 2 more cycles, bitwise: the final checkpoint.bin
/// and correlations.csv are byte-identical and segments.csv and observables.csv of the two legs
/// concatenate to the uninterrupted files. The checkpoint has the M7a layout (version-2 part of
/// the M6 length, then the two length-prefixed correlation buffers).
#[test]
fn pa_cmd_restart_at_a_cycle_end_is_bitwise() {
    let full_t = set(&with(DYN_PACMD), "sampling", "n_steps", "72");
    let half_t = set(&full_t, "sampling", "n_steps", "36");
    let (full, leg1, leg2) = (tmp(), tmp(), tmp());
    run_in(&parse(&full_t), full.path(), 1);
    run_in(&parse(&half_t), leg1.path(), 1);
    pool(1)
        .install(|| {
            run_from_checkpoint(
                &parse(&full_t),
                &leg1.path().join("checkpoint.bin"),
                leg2.path(),
            )
        })
        .unwrap();
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
    let raw = bytes(&full.path().join("checkpoint.bin"));
    let (bvv, bxx) = trailing_buffers(&raw, v2_len(4, 8, None));
    assert_eq!(
        CentroidCorrelation::from_bytes(&bvv).unwrap().n_segments(),
        4
    );
    assert_eq!(
        CentroidCorrelation::from_bytes(&bxx).unwrap().n_segments(),
        4
    );
}

/// |a − b| ≤ tol·max(|b|, scale).
fn close_s(a: f64, b: f64, tol: f64, scale: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(scale).max(1e-300),
        "{what}: {a:e} vs {b:e}"
    );
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Y-7, Y-6, Y-2 (amended), Y-3 … Y-5, D-2 (M7b test 7: "the CLI cycle equals a plain-loop
/// reference cycle … including segments.csv with H̃_P"): the PA-CMD run of the cluster equals
/// `reference_pacmd` (initial momenta from the amended Y-2 draw, Q32; Ω in the integrator
/// from the first equilibration step, PILE-L(τ_0)
/// decorrelation, amended redraw, `Trpmd { λ }` segments, one RNG): `RunSummary::correlations`
/// (C_vv, C_xx and their σ, D) to 1e-12 relative (σ 1e-9), the correlations.csv columns, and
/// segments.csv (one row per segment, h_start/h_end = H̃_P/N with the dynamical masses, by hand,
/// 1e-12). Also with decorrelation_steps = 0 and with λ = 0.25. (The bead kinetic energy in
/// segments.csv differs from H̃_P by far more than 1e-12 here: s_k ≈ 0.04–0.07.)
#[test]
fn pa_cmd_run_matches_reference_cycle() {
    let no_decorr = set(
        &set(
            &with(&DYN_PACMD.replace("decorrelation_steps = 6", "decorrelation_steps = 0")),
            "sampling",
            "n_steps",
            "36",
        ),
        "output",
        "checkpoint_every",
        "12",
    );
    let lam = with(&DYN_PACMD.replace("cmd_lambda = 1.0", "cmd_lambda = 0.25"));
    for toml in [with(DYN_PACMD), no_decorr, lam] {
        let cfg = parse(&toml);
        let dy = cfg.dynamics.clone().unwrap();
        let dir = tmp();
        let summary = run_in(&cfg, dir.path(), 1);
        let want = reference_pacmd(&cfg);
        let n_seg = want.cvv_seg.len();
        let c = summary
            .correlations
            .expect("correlations of a [dynamics] run");
        assert_eq!(c.n_segments, n_seg);
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
                &format!("C_vv σ[{k}]"),
            );
            close_s(got_xx[k], cxx[k], 1e-12, sxx, &format!("C_xx[{k}]"));
            close_s(
                got_xx_sem[k],
                cxx_sem[k],
                1e-9,
                sxs,
                &format!("C_xx σ[{k}]"),
            );
            close_s(c.diffusion[k], d[k], 1e-12, sd, &format!("D[{k}]"));
        }

        let (header, rows) = read_csv(&dir.path().join("correlations.csv"));
        assert_eq!(header, CORR_HEADER);
        assert_eq!(rows.len(), dy.n_lags);
        for (k, r) in rows.iter().enumerate() {
            let r: Vec<f64> = r.iter().map(|v| v.parse().unwrap()).collect();
            close_s(r[2], cvv[k], 1e-12, svv, "c_vv_au");
            close_s(r[4], d[k], 1e-12, sd, "d_au");
            close_s(r[6], cxx[k], 1e-12, sxx, "c_xx_bohr2");
        }

        let (header, seg) = read_csv(&dir.path().join("segments.csv"));
        assert_eq!(header, SEG_HEADER);
        assert_eq!(seg.len(), n_seg);
        let cyc = dy.decorrelation_steps + dy.segment_steps;
        for (s, (r, &(h0, h1))) in seg.iter().zip(&want.h).enumerate() {
            assert_eq!(r[0].parse::<usize>().unwrap(), s, "segment numbering");
            assert_eq!(
                r[1].parse::<usize>().unwrap(),
                (s + 1) * cyc,
                "segment step"
            );
            let (a, b): (f64, f64) = (r[2].parse().unwrap(), r[3].parse().unwrap());
            close_s(a, h0, 1e-12, 0.0, &format!("h_start_eh[{s}] (H̃_P/N)"));
            close_s(b, h1, 1e-12, 0.0, &format!("h_end_eh[{s}] (H̃_P/N)"));
        }
    }
}

/// MILESTONE_7 M7b decision 3 ("Without Ω every M7a rule and result is unchanged, bitwise"):
/// the M7a TRPMD and RPMD cluster runs write checkpoint.bin, segments.csv and correlations.csv
/// whose FNV-1a equal those written by `pimd-rs` at e6008c9 for the same configs.
#[test]
fn m7a_runs_unchanged_since_e6008c9() {
    for (dynamics, want) in [
        (DYN_TRPMD, CLUSTER_TRPMD_FNV1A_E6008C9),
        (DYN_RPMD, CLUSTER_RPMD_FNV1A_E6008C9),
    ] {
        let dir = tmp();
        run_in(&parse(&with(dynamics)), dir.path(), 1);
        for (f, w) in ["checkpoint.bin", "segments.csv", "correlations.csv"]
            .iter()
            .zip(want)
        {
            assert_eq!(fnv1a(&bytes(&dir.path().join(f))), w, "{f}");
        }
    }
}
