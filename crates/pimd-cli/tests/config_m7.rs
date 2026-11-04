//! `[dynamics]` config tests (MILESTONE_7 required test 8; "Config format", "Clarifications").
//!
//! Expected values: the brief's field rules, written out per case; the `config hash` of configs
//! without `[dynamics]` as printed by `pimd-rs` built at cf79a4d (recorded in `dyn_common`).
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod dyn_common;

use common::{BASE, parse, set, text, tmp};
use dyn_common::{BASE_HASH_CF79A4D, CLUSTER, CLUSTER_HASH_CF79A4D, DYN_RPMD, DYN_TRPMD};
use pimd_cli::{CliError, Config, DynamicsConfig, DynamicsMethod, run, run_from_checkpoint};

fn with(dynamics: &str) -> String {
    format!("{CLUSTER}{dynamics}")
}

/// MILESTONE_7 "Config format": a valid TRPMD section parses to the given fields; `trpmd_lambda`
/// defaults to 0.5; RPMD parses to `DynamicsMethod::Rpmd`; `decorrelation_steps = 0` is allowed;
/// n_lags = L = segment_steps/corr_stride + 1 is allowed; the rest of the config is the same as
/// without the section; no section gives `dynamics: None`.
#[test]
fn valid_sections_parse() {
    let cfg = parse(&with(DYN_TRPMD));
    assert_eq!(
        cfg.dynamics,
        Some(DynamicsConfig {
            method: DynamicsMethod::Trpmd { lambda: 0.5 },
            decorrelation_steps: 6,
            segment_steps: 12,
            corr_stride: 3,
            n_lags: 4,
        })
    );
    let plain = parse(CLUSTER);
    assert_eq!(plain.dynamics, None);
    assert_eq!(cfg.system, plain.system);
    assert_eq!(cfg.sampling, plain.sampling);
    assert_eq!(cfg.potential, plain.potential);
    assert_eq!(cfg.output, plain.output);

    let cfg = parse(&with(&DYN_TRPMD.replace("trpmd_lambda = 0.5\n", "")));
    assert_eq!(
        cfg.dynamics.unwrap().method,
        DynamicsMethod::Trpmd { lambda: 0.5 },
        "default trpmd_lambda"
    );
    let cfg = parse(&with(
        &DYN_TRPMD.replace("trpmd_lambda = 0.5", "trpmd_lambda = 1.25"),
    ));
    assert_eq!(
        cfg.dynamics.unwrap().method,
        DynamicsMethod::Trpmd { lambda: 1.25 }
    );
    let cfg = parse(&with(DYN_RPMD));
    assert_eq!(cfg.dynamics.clone().unwrap().method, DynamicsMethod::Rpmd);

    // decorrelation_steps = 0: cycle 12, n_steps 54 is not a multiple; use 48 and checkpoints
    // every 12.
    let t = set(
        &set(
            &with(&DYN_RPMD.replace("decorrelation_steps = 6", "decorrelation_steps = 0")),
            "sampling",
            "n_steps",
            "48",
        ),
        "output",
        "checkpoint_every",
        "12",
    );
    assert_eq!(parse(&t).dynamics.unwrap().decorrelation_steps, 0);
    // n_lags = L = 12/3 + 1 = 5.
    let cfg = parse(&with(&DYN_RPMD.replace("n_lags = 4", "n_lags = 5")));
    assert_eq!(cfg.dynamics.unwrap().n_lags, 5);
}

/// `InvalidValue` naming `what`.
fn assert_invalid(toml: &str, what: &str) {
    match Config::from_toml_str(toml) {
        Err(CliError::InvalidValue { what: w, .. }) => assert_eq!(w, what, "{toml}"),
        other => panic!("expected InvalidValue naming {what}, got {other:?}"),
    }
}

/// An error that names `key`: `InvalidValue` with `what == "dynamics.<key>"`, or `Parse` whose
/// message contains `<key>` (the brief says only "errors naming the `dynamics.<key>`" for these
/// cases; QUESTIONS Q29).
fn assert_names_key(toml: &str, key: &str) {
    match Config::from_toml_str(toml) {
        Err(CliError::InvalidValue { what, .. }) => assert_eq!(what, format!("dynamics.{key}")),
        Err(CliError::Parse(msg)) => assert!(msg.contains(key), "{msg:?} does not name {key}"),
        other => panic!("expected an error naming dynamics.{key}, got {other:?}"),
    }
}

/// MILESTONE_7 "Clarifications" (production with `[dynamics]`): n_steps must be a multiple of the
/// cycle c = decorrelation_steps + segment_steps (= 18) and checkpoint_every a multiple of c;
/// violations are `InvalidValue` naming "sampling.n_steps" / "output.checkpoint_every".
#[test]
fn cycle_multiples_enforced() {
    let base = with(DYN_TRPMD);
    assert_invalid(&set(&base, "sampling", "n_steps", "45"), "sampling.n_steps");
    assert_invalid(&set(&base, "sampling", "n_steps", "17"), "sampling.n_steps");
    assert_invalid(
        &set(&base, "output", "checkpoint_every", "9"),
        "output.checkpoint_every",
    );
    assert_invalid(
        &set(&base, "output", "checkpoint_every", "20"),
        "output.checkpoint_every",
    );
    // Multiples are fine (n_steps = 0 is a multiple too).
    parse(&set(&base, "output", "checkpoint_every", "36"));
    parse(&set(&base, "sampling", "n_steps", "0"));
}

/// MILESTONE_7 "Config format" / "Clarifications" (field rules): segment_steps ≥ 1 and a multiple
/// of corr_stride, corr_stride ≥ 1, 1 ≤ n_lags ≤ segment_steps/corr_stride + 1, trpmd_lambda > 0
/// and finite: each violation is `InvalidValue` naming "dynamics.<key>".
#[test]
fn field_ranges_enforced() {
    let r = |a: &str, b: &str| with(&DYN_TRPMD.replace(a, b));
    assert_invalid(&r("n_lags = 4", "n_lags = 6"), "dynamics.n_lags");
    assert_invalid(&r("n_lags = 4", "n_lags = 0"), "dynamics.n_lags");
    // segment_steps = 12 is not a multiple of corr_stride = 5 (n_lags 2 ≤ 12/5 + 1 stays valid).
    let t = with(
        &DYN_TRPMD
            .replace("corr_stride = 3", "corr_stride = 5")
            .replace("n_lags = 4", "n_lags = 2"),
    );
    assert_invalid(&t, "dynamics.segment_steps");
    assert_invalid(
        &r("corr_stride = 3", "corr_stride = 0"),
        "dynamics.corr_stride",
    );
    // segment_steps = 0 (cycle 6: n_steps 54 and checkpoint 18 are multiples; n_lags 1 ≤ 0/3 + 1).
    let t = with(
        &DYN_TRPMD
            .replace("segment_steps = 12", "segment_steps = 0")
            .replace("n_lags = 4", "n_lags = 1"),
    );
    assert_invalid(&t, "dynamics.segment_steps");
    for bad in ["0.0", "-0.5", "nan", "inf", "-inf"] {
        assert_invalid(
            &r("trpmd_lambda = 0.5", &format!("trpmd_lambda = {bad}")),
            "dynamics.trpmd_lambda",
        );
    }
}

/// MILESTONE_7 "Clarifications": `trpmd_lambda` with method = "rpmd", an unknown method, and
/// unknown keys in `[dynamics]` are errors naming the key.
#[test]
fn foreign_and_unknown_keys_rejected() {
    assert_names_key(
        &with(&DYN_RPMD.replace("method = \"rpmd\"", "method = \"rpmd\"\ntrpmd_lambda = 0.5")),
        "trpmd_lambda",
    );
    assert_names_key(
        &with(&format!("{DYN_TRPMD}cmd_omega_cm = 100.0\n")),
        "cmd_omega_cm",
    );
    assert_names_key(&with(&format!("{DYN_RPMD}n_lag = 3\n")), "n_lag");
    assert!(matches!(
        Config::from_toml_str(&with(&DYN_RPMD.replace("\"rpmd\"", "\"pa-cmd\""))),
        Err(CliError::Parse(_) | CliError::InvalidValue { .. })
    ));
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

/// MILESTONE_7 "No change without `[dynamics]`": for configs without the section the `config hash`
/// line of run.log equals the value printed by `pimd-rs` built at cf79a4d (M6), for the gas-phase
/// cluster and for the periodic M4 test system.
#[test]
fn hash_without_section_is_unchanged_since_m6() {
    for (toml, want) in [(CLUSTER, CLUSTER_HASH_CF79A4D), (BASE, BASE_HASH_CF79A4D)] {
        let dir = tmp();
        run(&parse(toml), dir.path()).unwrap();
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

/// MILESTONE_7 "No change without `[dynamics]`" / "Checkpoint block" (the M5 hash test pattern):
/// every `[dynamics]` field enters the config hash. A checkpoint of the TRPMD cluster run restarts
/// under the same config, but under a config that differs in any one `[dynamics]` field (method,
/// trpmd_lambda, decorrelation_steps, segment_steps, corr_stride, n_lags; n_steps and
/// checkpoint_every adjusted to stay valid) `run_from_checkpoint` is `CliError::Checkpoint`; so is
/// restarting it without `[dynamics]`, and restarting a run without the section under a config with
/// it.
#[test]
fn every_dynamics_field_is_hashed() {
    let base = with(DYN_TRPMD);
    let dir = one_cycle(&base);
    let ckpt = dir.path().join("checkpoint.bin");
    let out = tmp();
    // Positive control: same config, longer run.
    run_from_checkpoint(
        &parse(&set(&base, "sampling", "n_steps", "36")),
        &ckpt,
        out.path(),
    )
    .unwrap();
    let r = |a: &str, b: &str| DYN_TRPMD.replace(a, b);
    // (section, n_steps, checkpoint_every) for each changed field.
    let variants = [
        (DYN_RPMD.to_string(), "36", "18"),
        (r("trpmd_lambda = 0.5", "trpmd_lambda = 0.7"), "36", "18"),
        (
            r("decorrelation_steps = 6", "decorrelation_steps = 3"),
            "30",
            "15",
        ),
        (r("segment_steps = 12", "segment_steps = 9"), "30", "15"),
        (r("corr_stride = 3", "corr_stride = 4"), "36", "18"),
        (r("n_lags = 4", "n_lags = 3"), "36", "18"),
    ];
    for (section, n_steps, every) in variants {
        let t = set(
            &set(&with(&section), "sampling", "n_steps", n_steps),
            "output",
            "checkpoint_every",
            every,
        );
        let out = tmp();
        match run_from_checkpoint(&parse(&t), &ckpt, out.path()) {
            Err(CliError::Checkpoint(_)) => {}
            other => panic!("changed [dynamics] {section} restarted: {other:?}"),
        }
    }
    // Without the section.
    let out = tmp();
    assert!(matches!(
        run_from_checkpoint(
            &parse(&set(CLUSTER, "sampling", "n_steps", "36")),
            &ckpt,
            out.path()
        ),
        Err(CliError::Checkpoint(_))
    ));
    // And a checkpoint without the section under a config with it.
    let plain = one_cycle(CLUSTER);
    let out = tmp();
    assert!(matches!(
        run_from_checkpoint(
            &parse(&set(&base, "sampling", "n_steps", "36")),
            &plain.path().join("checkpoint.bin"),
            out.path()
        ),
        Err(CliError::Checkpoint(_))
    ));
}
