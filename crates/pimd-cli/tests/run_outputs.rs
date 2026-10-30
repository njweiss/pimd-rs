//! Run driver outputs (MILESTONE_4 "Outputs", "Required tests: Smoke, Reproducibility"; SPEC
//! D-2, D-3, U-3, E-4, E-7). File names, the CSV headers, the recording rule (s % stride == 0,
//! production only), the `{}` number format and the unit columns are checked against the brief;
//! byte-identical outputs across runs and thread counts against D-3.

mod common;

use std::path::Path;

use common::{
    BASE, EV_PER_HARTREE, OBS_HEADER, bytes, observables, parse, pool, read_csv, rel, remove, set,
    text, tmp,
};
use pimd_cli::{Config, RunSummary, run};
use pimd_core::units::{BOHR_IN_ANGSTROM, K_B_HARTREE_PER_KELVIN};

const FILES: [&str; 4] = ["observables.csv", "rdf.csv", "checkpoint.bin", "run.log"];

fn run_in(cfg: &Config, dir: &Path, threads: usize) -> RunSummary {
    pool(threads).install(|| run(cfg, dir).unwrap())
}

/// The 50-step smoke config (n_equil = 10, stride = 5).
fn smoke_config() -> Config {
    parse(&set(
        &set(BASE, "sampling", "n_steps", "50"),
        "sampling",
        "n_equil",
        "10",
    ))
}

/// "Required tests: Smoke": a 50-step run executes, writes observables.csv, rdf.csv,
/// checkpoint.bin and run.log, and reports n_steps = 50 and an RDF.
#[test]
fn smoke_run_writes_all_files() {
    let d = tmp();
    let out = d.path().join("out"); // not created beforehand: `run` creates it
    let s = run_in(&smoke_config(), &out, 1);
    for f in FILES {
        let p = out.join(f);
        assert!(p.is_file(), "{f} missing");
        assert!(!bytes(&p).is_empty(), "{f} empty");
    }
    assert_eq!(s.n_steps, 50);
    assert!(s.mean_e_prim.is_finite() && s.mean_e_vir.is_finite());
    let (r, g) = s.rdf.expect("RDF configured");
    assert_eq!((r.len(), g.len()), (20, 20));
    assert!(g.iter().all(|v| v.is_finite() && *v >= 0.0));
}

/// "Outputs" / "Observables": the header is exactly the brief's; one row per recorded step
/// s = stride, 2·stride, … ≤ n_steps (production numbering, equilibration records nothing); each
/// number is written with Rust's `{}` (re-formatting the parsed value gives the same text).
#[test]
fn observables_format_and_recording_rule() {
    let d = tmp();
    // n_steps = 47 is not a multiple of stride 5: rows at 5, 10, …, 45 (9 rows).
    let cfg = parse(&set(BASE, "sampling", "n_steps", "47"));
    run_in(&cfg, d.path(), 1);
    let (header, rows) = read_csv(&d.path().join("observables.csv"));
    assert_eq!(header, OBS_HEADER);
    let steps: Vec<String> = rows.iter().map(|r| r[0].clone()).collect();
    let want: Vec<String> = (1..=9).map(|k| (5 * k).to_string()).collect();
    assert_eq!(steps, want, "recorded steps");
    for row in &rows {
        assert_eq!(row.len(), 9, "row {row:?}");
        for v in &row[1..] {
            let x: f64 = v.parse().unwrap_or_else(|_| panic!("not a number: {v}"));
            assert!(x.is_finite());
            assert_eq!(format!("{x}"), *v, "not written with {{}}");
        }
    }
}

/// U-3, E-4 ("Outputs"): the unit columns are the Eh values converted: *_mev = Eh × 27211.386245988
/// (CODATA 2018; QUESTIONS Q13), *_k = Eh / k_B; to 1e-12 relative. T_est is positive and V < 0
/// for this bound liquid-like system; E_prim and E_vir are each V̄ plus a kinetic estimator, so
/// they exceed V on average.
#[test]
fn unit_columns_consistent() {
    let d = tmp();
    run_in(&smoke_config(), d.path(), 1);
    let rows = observables(d.path());
    assert_eq!(rows.len(), 10);
    for r in &rows {
        let (ep, ev) = (r[1], r[2]);
        assert!(
            rel(r[5], ep * EV_PER_HARTREE * 1000.0, 1e-300) <= 1e-12,
            "e_prim_mev {r:?}"
        );
        assert!(
            rel(r[6], ev * EV_PER_HARTREE * 1000.0, 1e-300) <= 1e-12,
            "e_vir_mev {r:?}"
        );
        assert!(
            rel(r[7], ep / K_B_HARTREE_PER_KELVIN, 1e-300) <= 1e-12,
            "e_prim_k {r:?}"
        );
        assert!(
            rel(r[8], ev / K_B_HARTREE_PER_KELVIN, 1e-300) <= 1e-12,
            "e_vir_k {r:?}"
        );
        assert!(r[4] > 0.0, "t_est {r:?}");
        assert!(r[3] < 0.0, "v {r:?}");
    }
    let n = rows.len() as f64;
    let mean = |c: usize| rows.iter().map(|r| r[c]).sum::<f64>() / n;
    assert!(mean(1) > mean(3) && mean(2) > mean(3));
}

/// E-4 ("Observables"): `mean_e_prim` / `mean_e_vir` are the averages of the recorded per-atom
/// values (the CSV columns e_prim_eh / e_vir_eh), to 1e-12 relative.
#[test]
fn summary_means_are_column_means() {
    let d = tmp();
    let s = run_in(&smoke_config(), d.path(), 1);
    let rows = observables(d.path());
    let n = rows.len() as f64;
    let mp = rows.iter().map(|r| r[1]).sum::<f64>() / n;
    let mv = rows.iter().map(|r| r[2]).sum::<f64>() / n;
    assert!(
        rel(s.mean_e_prim, mp, 1e-300) <= 1e-12,
        "{} vs {mp}",
        s.mean_e_prim
    );
    assert!(
        rel(s.mean_e_vir, mv, 1e-300) <= 1e-12,
        "{} vs {mv}",
        s.mean_e_vir
    );
}

/// E-7 ("Outputs"): rdf.csv has header `r_bohr,r_angstrom,g` and one row per bin; r_bohr are the
/// bin centres (b + ½)·r_max/n_bins, r_angstrom = r_bohr × 0.529177210903, and the columns equal
/// `RunSummary::rdf` exactly (shortest round-trip format).
#[test]
fn rdf_file_matches_summary() {
    let d = tmp();
    let cfg = smoke_config();
    let s = run_in(&cfg, d.path(), 1);
    let (r, g) = s.rdf.unwrap();
    let (header, rows) = read_csv(&d.path().join("rdf.csv"));
    assert_eq!(header, "r_bohr,r_angstrom,g");
    assert_eq!(rows.len(), 20);
    let rmax = cfg.output.rdf.as_ref().unwrap().r_max;
    for (b, row) in rows.iter().enumerate() {
        let v: Vec<f64> = row.iter().map(|x| x.parse().unwrap()).collect();
        let centre = (b as f64 + 0.5) * rmax / 20.0;
        assert!(
            rel(v[0], centre, 1e-300) <= 1e-14,
            "bin {b} centre {}",
            v[0]
        );
        assert_eq!(v[0].to_bits(), r[b].to_bits(), "bin {b} r");
        assert!(
            rel(v[1], v[0] * BOHR_IN_ANGSTROM, 1e-300) <= 1e-15,
            "bin {b} Å"
        );
        assert_eq!(v[2].to_bits(), g[b].to_bits(), "bin {b} g");
    }
}

/// "Config format": without RDF keys `RunSummary::rdf` is `None` and the run still succeeds.
#[test]
fn no_rdf_configured() {
    let d = tmp();
    let toml = remove(&remove(BASE, "rdf_bins"), "rdf_rmax_angstrom");
    let s = run_in(&parse(&toml), d.path(), 1);
    assert!(s.rdf.is_none());
    assert!(d.path().join("observables.csv").is_file());
    assert!(d.path().join("checkpoint.bin").is_file());
}

/// "Outputs": run.log records the seed (and echoes the config).
#[test]
fn run_log_records_seed() {
    let d = tmp();
    run_in(&smoke_config(), d.path(), 1);
    let log = text(&d.path().join("run.log"));
    assert!(log.contains("12345"), "seed not in run.log:\n{log}");
}

/// D-2, D-3 ("Reproducibility"): the same config run twice gives byte-identical output files.
#[test]
fn same_config_identical_bytes() {
    let (a, b) = (tmp(), tmp());
    let cfg = smoke_config();
    run_in(&cfg, a.path(), 1);
    run_in(&cfg, b.path(), 1);
    for f in FILES {
        assert_eq!(
            bytes(&a.path().join(f)),
            bytes(&b.path().join(f)),
            "{f} differs"
        );
    }
}

/// D-2 ("Reproducibility"): a different seed gives different observables and checkpoint.
#[test]
fn different_seed_differs() {
    let (a, b) = (tmp(), tmp());
    run_in(&smoke_config(), a.path(), 1);
    let other = parse(&set(
        &set(
            &set(BASE, "sampling", "n_steps", "50"),
            "sampling",
            "n_equil",
            "10",
        ),
        "sampling",
        "seed",
        "12346",
    ));
    run_in(&other, b.path(), 1);
    for f in ["observables.csv", "checkpoint.bin"] {
        assert_ne!(
            bytes(&a.path().join(f)),
            bytes(&b.path().join(f)),
            "{f} equal"
        );
    }
}

/// D-3: every output file is byte-identical between a 1-thread and a 4-thread pool, for each
/// normal-mode setting (matrix and fft); a run without an explicit pool matches too.
#[test]
fn identical_across_thread_counts_per_transform() {
    for nm in ["\"matrix\"", "\"fft\""] {
        let cfg = parse(&set(
            &set(BASE, "sampling", "n_steps", "50"),
            "sampling",
            "normal_modes",
            nm,
        ));
        let (a, b, c) = (tmp(), tmp(), tmp());
        run_in(&cfg, a.path(), 1);
        run_in(&cfg, b.path(), 4);
        run(&cfg, c.path()).unwrap();
        for f in FILES {
            let x = bytes(&a.path().join(f));
            assert_eq!(
                x,
                bytes(&b.path().join(f)),
                "{nm}: {f} differs, 1 vs 4 threads"
            );
            assert_eq!(
                x,
                bytes(&c.path().join(f)),
                "{nm}: {f} differs, global pool"
            );
        }
    }
}

/// RP-4, T-1 (transform option): over 50 steps (N = 32, P = 4) the fft and matrix settings give
/// the same observables to 1e-9 relative (same draws, transforms equal to ~1e-15; rounding
/// differences then grow along the LJ trajectory: measured 6e-14 at 50 steps, and below 1e-13
/// up to 2000 steps, with the reference implementation's direct-DFT transform).
#[test]
fn fft_and_matrix_runs_agree() {
    let (a, b) = (tmp(), tmp());
    let toml = set(BASE, "sampling", "n_steps", "50");
    run_in(&parse(&toml), a.path(), 1);
    run_in(
        &parse(&set(&toml, "sampling", "normal_modes", "\"fft\"")),
        b.path(),
        1,
    );
    let (ra, rb) = (observables(a.path()), observables(b.path()));
    assert_eq!(ra.len(), rb.len());
    for (x, y) in ra.iter().zip(&rb) {
        assert_eq!(x[0], y[0]);
        for c in 1..9 {
            assert!(
                rel(y[c], x[c], 1e-300) <= 1e-9,
                "column {c}: {} vs {}",
                y[c],
                x[c]
            );
        }
    }
}

/// "Binary": `pimd-rs <config.toml> --out <dir>` writes the same files as `run`, byte for byte,
/// and `--restart <checkpoint>` continues like `run_from_checkpoint`; a bad config exits non-zero.
#[test]
fn binary_matches_library() {
    let exe = env!("CARGO_BIN_EXE_pimd-rs");
    let d = tmp();
    let short = set(BASE, "sampling", "n_steps", "20");
    let cfg_path = d.path().join("short.toml");
    std::fs::write(&cfg_path, &short).unwrap();
    let full_path = d.path().join("full.toml");
    std::fs::write(&full_path, BASE).unwrap();
    let (bin1, bin2, lib1, lib2) = (
        d.path().join("bin1"),
        d.path().join("bin2"),
        d.path().join("lib1"),
        d.path().join("lib2"),
    );
    let status = std::process::Command::new(exe)
        .arg(&cfg_path)
        .arg("--out")
        .arg(&bin1)
        .env("RAYON_NUM_THREADS", "2")
        .status()
        .unwrap();
    assert!(status.success());
    let status = std::process::Command::new(exe)
        .arg(&full_path)
        .arg("--out")
        .arg(&bin2)
        .arg("--restart")
        .arg(bin1.join("checkpoint.bin"))
        .env("RAYON_NUM_THREADS", "2")
        .status()
        .unwrap();
    assert!(status.success());
    run_in(&parse(&short), &lib1, 1);
    pool(1).install(|| {
        pimd_cli::run_from_checkpoint(&parse(BASE), &lib1.join("checkpoint.bin"), &lib2).unwrap()
    });
    for f in ["observables.csv", "rdf.csv", "checkpoint.bin"] {
        assert_eq!(bytes(&bin1.join(f)), bytes(&lib1.join(f)), "first leg: {f}");
        assert_eq!(
            bytes(&bin2.join(f)),
            bytes(&lib2.join(f)),
            "restart leg: {f}"
        );
    }
    let bad = d.path().join("bad.toml");
    std::fs::write(&bad, set(BASE, "sampling", "n_beads", "0")).unwrap();
    let status = std::process::Command::new(exe)
        .arg(&bad)
        .arg("--out")
        .arg(d.path().join("bad"))
        .status()
        .unwrap();
    assert!(!status.success());
}
