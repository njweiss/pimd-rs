//! `potential.beads_per_graph` and `system.positions_xyz` with kind = "mace" (MILESTONE_8 "M8a:
//! Config", required test 5; SPEC P-6, D-3 chunk-size sentence). Feature `torch`; the tiny test
//! model of `pimd-torch`.
//!
//! `beads_per_graph`: optional, ≥ 1, default all beads in one graph (`None`); 0 is
//! `InvalidValue("potential.beads_per_graph")`. Hash (brief): the effective chunk size min(c, P)
//! is appended only when it is < P, so configs without the key, or with c ≥ P, hash as on main.
//! The main values are the `config hash` lines printed by `pimd-rs` built at main = 3e6a5bd (M7b
//! merged; no crate changes up to the M8a brief commit 027b31a) for `mace_water_toml(tiny export,
//! 10)` with n_beads = 4 and n_beads = 1 (recorded during test authoring, 2026-11-05, with a
//! throwaway test in a worktree of 3e6a5bd; the hash covers the model's bytes, not its path).
//! "Part of the config hash" is also checked as the M5–M7 tests do: a checkpoint restarts only
//! under a config with the same hash.
//!
//! `positions_xyz` with kind = "mace": the file's symbols must be known (`pimd_torch::
//! atomic_number`, the existing rule for `symbols`); the parsed config equals its inline
//! equivalent (symbols enter the hash for kind = "mace").
#![cfg(feature = "torch")]

mod common;
mod mace_common;

use std::path::Path;

use common::{parse, pool, set, text, tmp};
use mace_common::{mace_water_toml, tiny_export};
use pimd_cli::{CliError, Config, PotentialConfig, run, run_from_checkpoint};

/// `config hash` of `mace_water_toml(tiny export, ·)` (n_beads = 4) at main 3e6a5bd.
const HASH_P4_MAIN: &str = "0xb14892014363f1fd";
/// The same config with n_beads = 1, at main 3e6a5bd.
const HASH_P1_MAIN: &str = "0xfc9b4ad40c4fcee0";

fn err(toml: &str) -> CliError {
    match Config::from_toml_str(toml) {
        Ok(c) => panic!("config accepted: {c:?}\n{toml}"),
        Err(e) => e,
    }
}

fn base() -> String {
    mace_water_toml(&tiny_export(), 10)
}

fn with_c(toml: &str, c: usize) -> String {
    set(toml, "potential", "beads_per_graph", &c.to_string())
}

/// `beads_per_graph` of a mace config.
fn chunk(c: &Config) -> Option<usize> {
    match &c.potential {
        PotentialConfig::Mace {
            beads_per_graph, ..
        } => *beads_per_graph,
        other => panic!("expected kind = \"mace\", got {other:?}"),
    }
}

/// The `config hash = …` value of a run.log.
fn hash_of(dir: &Path) -> String {
    text(&dir.join("run.log"))
        .lines()
        .find(|l| l.starts_with("config hash = "))
        .expect("config hash line")
        .trim_start_matches("config hash = ")
        .to_string()
}

fn run_dir(toml: &str) -> tempfile::TempDir {
    let d = tmp();
    pool(1)
        .install(|| run(&parse(toml), d.path()))
        .unwrap_or_else(|e| panic!("run failed: {e:?}"));
    d
}

/// P-6 config (MILESTONE_8 test 5): `beads_per_graph` parses to `Some(c)` for c = 1, 3, 4, 8 and
/// 1000 (any c ≥ 1, also above P); without the key it is `None` (one graph of all beads), with
/// every other mace field unchanged.
#[test]
fn beads_per_graph_parses_and_defaults() {
    assert_eq!(chunk(&parse(&base())), None);
    for c in [1, 3, 4, 8, 1000] {
        let cfg = parse(&with_c(&base(), c));
        assert_eq!(chunk(&cfg), Some(c), "c = {c}");
        let mut plain = parse(&base());
        if let PotentialConfig::Mace {
            beads_per_graph, ..
        } = &mut plain.potential
        {
            *beads_per_graph = Some(c);
        }
        assert_eq!(cfg, plain, "c = {c}: only beads_per_graph differs");
    }
}

/// P-6 config (MILESTONE_8 test 5): `beads_per_graph = 0` → `InvalidValue("potential.
/// beads_per_graph")`; a negative or non-integer value is rejected (type error: `Parse`, or
/// `InvalidValue` naming the key).
#[test]
fn beads_per_graph_zero_and_bad_types() {
    match err(&with_c(&base(), 0)) {
        CliError::InvalidValue { what, .. } => assert_eq!(what, "potential.beads_per_graph"),
        other => panic!("expected InvalidValue(potential.beads_per_graph), got {other:?}"),
    }
    for v in ["-1", "2.5", "\"4\"", "true"] {
        match err(&set(&base(), "potential", "beads_per_graph", v)) {
            CliError::Parse(m) => assert!(m.contains("beads_per_graph"), "{v}: {m}"),
            CliError::InvalidValue { what, .. } => assert_eq!(what, "potential.beads_per_graph"),
            other => panic!("{v}: expected Parse/InvalidValue, got {other:?}"),
        }
    }
}

/// D-3, MILESTONE_8 "Config" hash (test 5): at P = 4 the config hash without the key, with c = 4
/// and with c = 7 equals main's value; c = 1, 2, 3 each change it (and differ from each other);
/// at P = 1, c = 1 and c = 2 hash as main.
#[test]
fn beads_per_graph_hash() {
    assert_eq!(
        hash_of(run_dir(&base()).path()),
        HASH_P4_MAIN,
        "no key, P = 4"
    );
    for c in [4, 7] {
        assert_eq!(
            hash_of(run_dir(&with_c(&base(), c)).path()),
            HASH_P4_MAIN,
            "c = {c} >= P = 4 must hash as main"
        );
    }
    let mut seen = vec![HASH_P4_MAIN.to_string()];
    for c in [1, 2, 3] {
        let h = hash_of(run_dir(&with_c(&base(), c)).path());
        assert!(!seen.contains(&h), "c = {c} < P: hash {h} not distinct");
        seen.push(h);
    }
    let p1 = set(&base(), "sampling", "n_beads", "1");
    assert_eq!(hash_of(run_dir(&p1).path()), HASH_P1_MAIN, "no key, P = 1");
    for c in [1, 2] {
        assert_eq!(
            hash_of(run_dir(&with_c(&p1, c)).path()),
            HASH_P1_MAIN,
            "P = 1, c = {c}"
        );
    }
}

/// D-3 (chunk size part of the config hash), MILESTONE_8 test 5: a checkpoint of a c = 2 run
/// restarts under c = 2, but not under c = 3, c = 4 (= P) or without the key
/// (`CliError::Checkpoint`); a checkpoint of a run without the key restarts under c = 4 and c = 7
/// (c ≥ P hashes as no chunking) but not under c = 3.
#[test]
fn beads_per_graph_restart_follows_hash() {
    let c2 = run_dir(&with_c(&base(), 2));
    let plain = run_dir(&base());
    let restart = |toml: &str, from: &Path| {
        let out = tmp();
        let t = set(toml, "sampling", "n_steps", "20");
        pool(1).install(|| {
            run_from_checkpoint(&parse(&t), &from.join("checkpoint.bin"), out.path()).map(|_| ())
        })
    };
    restart(&with_c(&base(), 2), c2.path()).expect("same c restarts");
    for (t, what) in [
        (with_c(&base(), 3), "c = 3"),
        (with_c(&base(), 4), "c = 4"),
        (base(), "no key"),
    ] {
        assert!(
            matches!(restart(&t, c2.path()), Err(CliError::Checkpoint(_))),
            "c = 2 checkpoint restarted under {what}"
        );
    }
    for c in [4, 7] {
        restart(&with_c(&base(), c), plain.path())
            .unwrap_or_else(|e| panic!("c = {c} >= P: restart of a plain run failed: {e:?}"));
    }
    assert!(matches!(
        restart(&with_c(&base(), 3), plain.path()),
        Err(CliError::Checkpoint(_))
    ));
}

/// The water molecule of `mace_water_toml` as an xyz file with the given symbols.
fn water_xyz(dir: &Path, symbols: [&str; 3]) -> std::path::PathBuf {
    let x = [
        ["0.0", "0.0", "0.1173"],
        ["0.0", "0.7572", "-0.4692"],
        ["0.0", "-0.7572", "-0.4692"],
    ];
    let mut s = String::from("3\nwater, M5 start geometry\n");
    for (sym, c) in symbols.iter().zip(x) {
        s.push_str(&format!("{sym} {} {} {}\n", c[0], c[1], c[2]));
    }
    let p = dir.join("water.xyz");
    std::fs::write(&p, s).unwrap();
    p
}

/// `base()` with the inline positions and symbols replaced by `positions_xyz = path`.
fn xyz_config(path: &Path) -> String {
    let t = common::remove(&base(), "positions_angstrom");
    let t = common::remove(&t, "symbols");
    set(
        &t,
        "system",
        "positions_xyz",
        &format!("\"{}\"", path.display()),
    )
}

/// P-6, MILESTONE_8 "Config" (test 5): with kind = "mace" the xyz file's symbols are the model's
/// atoms: an xyz config equals the inline config (positions and symbols); a symbol unknown to
/// `atomic_number` ("Xx", "h") is rejected naming `system.symbols` or `system.positions_xyz`.
#[test]
fn mace_xyz_symbols() {
    let d = tmp();
    assert_eq!(
        parse(&xyz_config(&water_xyz(d.path(), ["O", "H", "H"]))),
        parse(&base())
    );
    for bad in ["Xx", "h"] {
        let t = xyz_config(&water_xyz(d.path(), ["O", bad, "H"]));
        match err(&t) {
            CliError::InvalidValue { what, .. } => assert!(
                what == "system.symbols" || what == "system.positions_xyz",
                "{bad}: named {what}"
            ),
            other => panic!("{bad}: expected InvalidValue, got {other:?}"),
        }
    }
}
