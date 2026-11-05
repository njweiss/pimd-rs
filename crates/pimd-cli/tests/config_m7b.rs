//! `[dynamics] method = "pa-cmd"` config tests (MILESTONE_7 M7b required test 6; "Config format
//! (addition)", "Clarifications"; SPEC Y-6, Y-7, U-2, U-3).
//!
//! Expected values: the brief's field rules written out per case; Ω = cmd_omega_cm / 219474.6313632
//! (U-2, written in `pacmd_common`); `config hash` lines recorded with `pimd-rs` built at e6008c9
//! (M7a configs) and cf79a4d (no `[dynamics]`).
#![allow(clippy::needless_range_loop)]

mod common;
mod dyn_common;
mod pacmd_common;

use common::{BASE, parse, set, text, tmp};
use dyn_common::{BASE_HASH_CF79A4D, CLUSTER, CLUSTER_HASH_CF79A4D, DYN_RPMD, DYN_TRPMD};
use pacmd_common::{
    CLUSTER_RPMD_HASH_E6008C9, CLUSTER_TRPMD_HASH_E6008C9, CM_PER_HARTREE, DYN_PACMD,
};
use pimd_cli::{CliError, Config, DynamicsConfig, DynamicsMethod, run, run_from_checkpoint};

fn with(dynamics: &str) -> String {
    format!("{CLUSTER}{dynamics}")
}

fn pa(a: &str, b: &str) -> String {
    with(&DYN_PACMD.replace(a, b))
}

/// The (Ω, λ) of a parsed PA-CMD config.
fn omega_lambda(toml: &str) -> (f64, f64) {
    match parse(toml).dynamics.unwrap().method {
        DynamicsMethod::PaCmd { omega, lambda } => (omega, lambda),
        other => panic!("expected PaCmd, got {other:?}"),
    }
}

/// Y-6, Y-7, U-2, U-3 (M7b "Config format", decision 5): `method = "pa-cmd"` parses to
/// `DynamicsMethod::PaCmd { omega, lambda }` with Ω = cmd_omega_cm / HARTREE_IN_WAVENUMBERS
/// (angular frequency, ħ = 1: 412.5 cm⁻¹ → 1.8795e-3 Eh/ħ; no 2π) to 1e-15 relative and
/// λ = cmd_lambda exactly; the other fields as written; the rest of the config as without the
/// section. Also Ω = 16 000 cm⁻¹ (Rossi et al. 2014's value) and λ = 0.005 (Castro et al.'s
/// range) parse.
#[test]
fn pa_cmd_parses() {
    let cfg = parse(&with(DYN_PACMD));
    let dy = cfg.dynamics.clone().unwrap();
    let DynamicsMethod::PaCmd { omega, lambda } = dy.method else {
        panic!("expected PaCmd, got {:?}", dy.method);
    };
    let want = 412.5 / CM_PER_HARTREE;
    assert!(
        (omega - want).abs() <= 1e-15 * want,
        "Ω = {omega:e} vs {want:e}"
    );
    assert_eq!(lambda, 1.0);
    assert_eq!(
        dy,
        DynamicsConfig {
            method: DynamicsMethod::PaCmd { omega, lambda },
            decorrelation_steps: 6,
            segment_steps: 12,
            corr_stride: 3,
            n_lags: 4,
        }
    );
    let plain = parse(CLUSTER);
    assert_eq!(cfg.system, plain.system);
    assert_eq!(cfg.sampling, plain.sampling);
    assert_eq!(cfg.potential, plain.potential);
    assert_eq!(cfg.output, plain.output);

    let (w, l) = omega_lambda(
        &pa("cmd_lambda = 1.0", "cmd_lambda = 0.005")
            .replace("cmd_omega_cm = 412.5", "cmd_omega_cm = 16000.0"),
    );
    let want = 16000.0 / CM_PER_HARTREE;
    assert!((w - want).abs() <= 1e-15 * want, "Ω = {w:e}");
    assert_eq!(l, 0.005);
}

/// `InvalidValue` naming `what`.
fn assert_invalid(toml: &str, what: &str) {
    match Config::from_toml_str(toml) {
        Err(CliError::InvalidValue { what: w, .. }) => assert_eq!(w, what, "{toml}"),
        other => panic!("expected InvalidValue naming {what}, got {other:?}"),
    }
}

/// An error naming `key` (Q29 rules): `InvalidValue` with `what == "dynamics.<key>"`, or `Parse`
/// whose message contains `<key>`.
fn assert_names_key(toml: &str, key: &str) {
    match Config::from_toml_str(toml) {
        Err(CliError::InvalidValue { what, .. }) => assert_eq!(what, format!("dynamics.{key}")),
        Err(CliError::Parse(msg)) => assert!(msg.contains(key), "{msg:?} does not name {key}"),
        other => panic!("expected an error naming dynamics.{key}, got {other:?}"),
    }
}

/// M7b "Clarifications": Ω and λ must be positive and finite, else
/// `InvalidValue("dynamics.cmd_omega_cm")` / `InvalidValue("dynamics.cmd_lambda")`; no check of
/// Ω·dt (a huge Ω, Ω·dt ≈ 40, parses).
#[test]
fn omega_and_lambda_ranges() {
    for bad in ["0.0", "-0.0", "-400.0", "nan", "inf", "-inf"] {
        assert_invalid(
            &pa("cmd_omega_cm = 412.5", &format!("cmd_omega_cm = {bad}")),
            "dynamics.cmd_omega_cm",
        );
        assert_invalid(
            &pa("cmd_lambda = 1.0", &format!("cmd_lambda = {bad}")),
            "dynamics.cmd_lambda",
        );
    }
    let (w, _) = omega_lambda(&pa("cmd_omega_cm = 412.5", "cmd_omega_cm = 1.0e5"));
    assert!(w > 0.4, "Ω·dt is not checked: Ω = {w}");
}

/// M7b "Clarifications" (Q29 rules): `cmd_omega_cm` and `cmd_lambda` are required with
/// "pa-cmd" (a missing one is an error naming it) and rejected with "rpmd" and "trpmd";
/// `trpmd_lambda` is rejected with "pa-cmd" (so a λ cannot be read from the TRPMD key).
#[test]
fn pa_cmd_keys_required_and_exclusive() {
    assert_names_key(&pa("cmd_omega_cm = 412.5\n", ""), "cmd_omega_cm");
    assert_names_key(&pa("cmd_lambda = 1.0\n", ""), "cmd_lambda");
    // trpmd_lambda instead of cmd_lambda (both rules apply; either key may be named), and in
    // addition to it.
    match Config::from_toml_str(&pa("cmd_lambda = 1.0", "trpmd_lambda = 1.0")) {
        Err(CliError::InvalidValue { what, .. }) => assert!(
            what == "dynamics.cmd_lambda" || what == "dynamics.trpmd_lambda",
            "{what}"
        ),
        Err(CliError::Parse(msg)) => assert!(
            msg.contains("cmd_lambda") || msg.contains("trpmd_lambda"),
            "{msg:?}"
        ),
        other => panic!("trpmd_lambda instead of cmd_lambda accepted: {other:?}"),
    }
    assert_names_key(
        &pa("cmd_lambda = 1.0", "cmd_lambda = 1.0\ntrpmd_lambda = 0.5"),
        "trpmd_lambda",
    );
    for (section, method) in [(DYN_RPMD, "rpmd"), (DYN_TRPMD, "trpmd")] {
        let m = format!("method = \"{method}\"");
        assert_names_key(
            &with(&section.replace(&m, &format!("{m}\ncmd_omega_cm = 412.5"))),
            "cmd_omega_cm",
        );
        assert_names_key(
            &with(&section.replace(&m, &format!("{m}\ncmd_lambda = 1.0"))),
            "cmd_lambda",
        );
    }
}

/// MILESTONE_7 "Clarifications" (cycle rules), unchanged for "pa-cmd": n_steps and
/// checkpoint_every must be multiples of the cycle (18), n_lags ≤ L, segment_steps a multiple of
/// corr_stride, each `InvalidValue` naming its key.
#[test]
fn cycle_rules_apply_to_pa_cmd() {
    let base = with(DYN_PACMD);
    assert_invalid(&set(&base, "sampling", "n_steps", "45"), "sampling.n_steps");
    assert_invalid(
        &set(&base, "output", "checkpoint_every", "9"),
        "output.checkpoint_every",
    );
    assert_invalid(&pa("n_lags = 4", "n_lags = 6"), "dynamics.n_lags");
    assert_invalid(
        &with(
            &DYN_PACMD
                .replace("corr_stride = 3", "corr_stride = 5")
                .replace("n_lags = 4", "n_lags = 2"),
        ),
        "dynamics.segment_steps",
    );
}

/// The `config hash = …` line of a run.log.
fn hash_line(dir: &std::path::Path) -> String {
    text(&dir.join("run.log"))
        .lines()
        .find(|l| l.starts_with("config hash = "))
        .expect("config hash line")
        .trim_start_matches("config hash = ")
        .to_string()
}

/// M7b "Clarifications" ("M7a configs hash as on main at e6008c9, and configs without
/// `[dynamics]` as at cf79a4d"): the run.log `config hash` of the M7a TRPMD and RPMD cluster
/// configs equals the value printed by `pimd-rs` at e6008c9, and of the cluster and the M4 system
/// without the section the value printed at cf79a4d.
#[test]
fn m7a_and_plain_hashes_unchanged() {
    for (toml, want) in [
        (with(DYN_TRPMD), CLUSTER_TRPMD_HASH_E6008C9),
        (with(DYN_RPMD), CLUSTER_RPMD_HASH_E6008C9),
        (CLUSTER.to_string(), CLUSTER_HASH_CF79A4D),
        (BASE.to_string(), BASE_HASH_CF79A4D),
    ] {
        let dir = tmp();
        run(&parse(&toml), dir.path()).unwrap();
        assert_eq!(hash_line(dir.path()), want);
    }
}

/// Runs `toml` one cycle (n_steps = 18) and returns the directory (checkpoint at step 18).
fn one_cycle(toml: &str) -> tempfile::TempDir {
    let dir = tmp();
    let t = set(toml, "sampling", "n_steps", "18");
    run(&parse(&t), dir.path()).unwrap();
    dir
}

/// M7b "Clarifications" (the config hash appends the method tag, Ω and λ for "pa-cmd"; the M5
/// hash test pattern): a checkpoint of the PA-CMD cluster run restarts under the same config
/// (positive control), but under a config with a different cmd_omega_cm, a different cmd_lambda,
/// TRPMD with the same λ, RPMD, or no `[dynamics]`, `run_from_checkpoint` is `CliError::Checkpoint`;
/// so is restarting a TRPMD (λ = 1) checkpoint under the PA-CMD config. Also: a PA-CMD run differs
/// from TRPMD with the same λ (Ω enters the integrator) in its run.log hash.
#[test]
fn omega_lambda_and_method_are_hashed() {
    let base = with(DYN_PACMD);
    let dir = one_cycle(&base);
    let ckpt = dir.path().join("checkpoint.bin");
    let out = tmp();
    run_from_checkpoint(
        &parse(&set(&base, "sampling", "n_steps", "36")),
        &ckpt,
        out.path(),
    )
    .unwrap();
    let trpmd1 = with(&DYN_TRPMD.replace("trpmd_lambda = 0.5", "trpmd_lambda = 1.0"));
    let variants = [
        pa("cmd_omega_cm = 412.5", "cmd_omega_cm = 412.75"),
        pa("cmd_lambda = 1.0", "cmd_lambda = 0.75"),
        trpmd1.clone(),
        with(DYN_RPMD),
        CLUSTER.to_string(),
    ];
    for t in variants {
        let out = tmp();
        match run_from_checkpoint(
            &parse(&set(&t, "sampling", "n_steps", "36")),
            &ckpt,
            out.path(),
        ) {
            Err(CliError::Checkpoint(_)) => {}
            other => panic!("changed config restarted a pa-cmd checkpoint: {other:?}\n{t}"),
        }
    }
    let tr = one_cycle(&trpmd1);
    let out = tmp();
    assert!(matches!(
        run_from_checkpoint(
            &parse(&set(&base, "sampling", "n_steps", "36")),
            &tr.path().join("checkpoint.bin"),
            out.path()
        ),
        Err(CliError::Checkpoint(_))
    ));
    assert_ne!(hash_line(dir.path()), hash_line(tr.path()));
}

/// M7b "Clarifications": run.log echoes Ω in cm⁻¹ and in Eh/ħ, and λ. Checked loosely (the
/// format is not specified): the log contains "412.5" (Ω in cm⁻¹), the `{}` rendering of the
/// parsed Ω in Eh/ħ, and "0.625" (λ, chosen so that its rendering is unambiguous).
#[test]
fn run_log_echoes_omega_and_lambda() {
    let t = pa("cmd_lambda = 1.0", "cmd_lambda = 0.625");
    let dir = one_cycle(&t);
    let (w, _) = omega_lambda(&t);
    let log = text(&dir.path().join("run.log"));
    assert!(log.contains("412.5"), "Ω in cm⁻¹ not echoed:\n{log}");
    assert!(
        log.contains(&format!("{w}")),
        "Ω = {w} Eh/ħ not echoed:\n{log}"
    );
    assert!(log.contains("0.625"), "λ not echoed:\n{log}");
}
