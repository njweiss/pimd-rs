//! Checkpoint / restart (MILESTONE_4 "Restart determinism", "Required tests: Restart"; SPEC D-2,
//! D-3). A run of 2K production steps must equal K steps → checkpoint → K more, bitwise: final
//! checkpoint, observables (the restarted file holds only rows after the checkpoint; header once
//! in the concatenation), RDF, and the summary averages (accumulators restored). N = 32, P = 4,
//! inside 1-thread and 4-thread pools. A config whose hash differs (any field of [system],
//! [potential], [sampling] except n_steps) or a corrupt file is `CliError::Checkpoint`.
//!
//! Assumption (recorded in the report): `checkpoint.bin` is a function of the simulation state
//! and the config hash only, so two runs that reach the same state write the same bytes.

mod common;

use std::path::Path;

use common::{BASE, OBS_HEADER, bytes, parse, pool, read_csv, set, text, tmp};
use pimd_cli::{CliError, Config, RunSummary, run, run_from_checkpoint};

/// Production rows of `observables.csv` (header checked, then dropped).
fn rows(dir: &Path) -> Vec<String> {
    let t = text(&dir.join("observables.csv"));
    let mut lines = t.lines();
    assert_eq!(lines.next(), Some(OBS_HEADER));
    lines
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

fn with_steps(n: usize) -> Config {
    parse(&set(BASE, "sampling", "n_steps", &n.to_string()))
}

fn assert_summary_bitwise(a: &RunSummary, b: &RunSummary, what: &str) {
    assert_eq!(a.n_steps, b.n_steps, "{what}: n_steps");
    assert_eq!(
        a.mean_e_prim.to_bits(),
        b.mean_e_prim.to_bits(),
        "{what}: mean_e_prim"
    );
    assert_eq!(
        a.mean_e_vir.to_bits(),
        b.mean_e_vir.to_bits(),
        "{what}: mean_e_vir"
    );
    let (ra, ga) = a.rdf.as_ref().unwrap();
    let (rb, gb) = b.rdf.as_ref().unwrap();
    assert_eq!(ra, rb, "{what}: rdf bins");
    let ga: Vec<u64> = ga.iter().map(|x| x.to_bits()).collect();
    let gb: Vec<u64> = gb.iter().map(|x| x.to_bits()).collect();
    assert_eq!(ga, gb, "{what}: rdf g");
}

/// Uninterrupted run of `total` steps vs `k` steps + checkpoint + the rest, in a pool of
/// `threads`. Returns the uninterrupted directory's final checkpoint bytes (for cross-thread
/// comparison).
fn check_restart(k: usize, total: usize, threads: usize) -> Vec<u8> {
    let (full, first, second) = (tmp(), tmp(), tmp());
    let what = format!("{k} + {} steps, {threads} threads", total - k);
    let (s_full, s_second) = pool(threads).install(|| {
        let s_full = run(&with_steps(total), full.path()).unwrap();
        let s_first = run(&with_steps(k), first.path()).unwrap();
        assert_eq!(s_first.n_steps, k);
        let s_second = run_from_checkpoint(
            &with_steps(total),
            &first.path().join("checkpoint.bin"),
            second.path(),
        )
        .unwrap();
        (s_full, s_second)
    });
    // Final state (and RNG position, accumulators): the end-of-run checkpoints agree byte for byte.
    let ck = bytes(&full.path().join("checkpoint.bin"));
    assert_eq!(
        ck,
        bytes(&second.path().join("checkpoint.bin")),
        "{what}: final checkpoint"
    );
    // Observables: first part + restarted part == uninterrupted; restarted rows all after k.
    let (r1, r2, rf) = (rows(first.path()), rows(second.path()), rows(full.path()));
    for r in &r2 {
        let step: usize = r.split(',').next().unwrap().parse().unwrap();
        assert!(
            step > k,
            "{what}: restarted file has row for step {step} ≤ {k}"
        );
    }
    let joined: Vec<String> = r1.into_iter().chain(r2).collect();
    assert_eq!(joined, rf, "{what}: observables concatenation");
    // RDF file and summary (accumulated over the whole production run).
    assert_eq!(
        read_csv(&full.path().join("rdf.csv")),
        read_csv(&second.path().join("rdf.csv")),
        "{what}: rdf.csv"
    );
    assert_summary_bitwise(&s_full, &s_second, &what);
    ck
}

/// D-2, D-3: 40 = 20 + 20 steps (checkpoint on a recording step, stride 5), bitwise, 1 thread.
#[test]
fn restart_bitwise_k_plus_k_one_thread() {
    check_restart(20, 40, 1);
}

/// D-2, D-3: as above in a 4-thread pool, and the 4-thread result equals the 1-thread result.
#[test]
fn restart_bitwise_k_plus_k_four_threads() {
    let a = check_restart(20, 40, 4);
    let b = check_restart(20, 40, 1);
    assert_eq!(a, b, "final checkpoint differs between 1 and 4 threads");
}

/// D-2, D-3: a checkpoint between recording steps (23 = 4·5 + 3, stride 5) and an odd remainder
/// (40 − 23 = 17): the stride phase continues, no row is lost or duplicated, still bitwise.
#[test]
fn restart_between_recording_steps() {
    check_restart(23, 40, 1);
    check_restart(23, 40, 4);
}

/// D-3: a chain of three legs (10 → 25 → 40) equals the uninterrupted run.
#[test]
fn restart_chain_of_three() {
    let (full, a, b, c) = (tmp(), tmp(), tmp(), tmp());
    pool(2).install(|| {
        let s_full = run(&with_steps(40), full.path()).unwrap();
        run(&with_steps(10), a.path()).unwrap();
        run_from_checkpoint(&with_steps(25), &a.path().join("checkpoint.bin"), b.path()).unwrap();
        let s = run_from_checkpoint(&with_steps(40), &b.path().join("checkpoint.bin"), c.path())
            .unwrap();
        assert_summary_bitwise(&s_full, &s, "chain");
    });
    let joined: Vec<String> = [a.path(), b.path(), c.path()]
        .iter()
        .flat_map(|d| rows(d))
        .collect();
    assert_eq!(joined, rows(full.path()));
    assert_eq!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&c.path().join("checkpoint.bin"))
    );
}

/// "Restart determinism": the config hash covers every field of [system], [potential] and
/// [sampling] except n_steps. Restarting a 10-step checkpoint with any one of them changed is
/// `CliError::Checkpoint` (including the normal-mode setting, MILESTONE_4 transform option).
#[test]
fn hash_mismatch_rejected_for_every_hashed_field() {
    let d = tmp();
    pool(1).install(|| run(&with_steps(10), d.path()).unwrap());
    let ck = d.path().join("checkpoint.bin");
    let base40 = set(BASE, "sampling", "n_steps", "40");
    for (section, key, value) in [
        ("system", "n_atoms", "31"),
        ("system", "mass_amu", "20.18"),
        ("system", "box_angstrom", "12.6"),
        ("system", "box_angstrom", "[12.5, 12.5, 12.6]"),
        ("potential", "epsilon_kelvin", "35.7"),
        ("potential", "sigma_angstrom", "2.75"),
        ("potential", "cutoff_sigma", "1.9"),
        ("potential", "skin_sigma", "0.2"),
        ("potential", "shift", "false"),
        ("sampling", "temperature_kelvin", "31.0"),
        ("sampling", "n_beads", "2"),
        ("sampling", "dt_fs", "0.9"),
        ("sampling", "tau0_fs", "90.0"),
        ("sampling", "n_equil", "21"),
        ("sampling", "stride", "2"),
        ("sampling", "seed", "12346"),
        ("sampling", "normal_modes", "\"fft\""),
    ] {
        let cfg = parse(&set(&base40, section, key, value));
        let out = tmp();
        let r = pool(1).install(|| run_from_checkpoint(&cfg, &ck, out.path()));
        match r {
            Err(CliError::Checkpoint(_)) => {}
            other => panic!("{section}.{key} = {value}: expected Checkpoint error, got {other:?}"),
        }
    }
}

/// "Restart determinism": the hash is over the parsed config, not the text, and excludes n_steps
/// and [output]: an explicit `normal_modes = "matrix"`, a three-value cubic box, comments and key
/// order, other n_steps and checkpoint_every all restart fine.
#[test]
fn hash_ignores_n_steps_output_and_formatting() {
    let d = tmp();
    pool(1).install(|| run(&with_steps(10), d.path()).unwrap());
    let ck = d.path().join("checkpoint.bin");
    let base40 = set(BASE, "sampling", "n_steps", "40");
    let reordered = base40
        .replace("seed = 12345", "")
        .replace("[sampling]", "[sampling]\n# comment\nseed = 12345");
    for toml in [
        base40.clone(),
        set(BASE, "sampling", "n_steps", "15"),
        set(&base40, "sampling", "normal_modes", "\"matrix\""),
        set(&base40, "system", "box_angstrom", "[12.5, 12.5, 12.5]"),
        set(&base40, "output", "checkpoint_every", "7"),
        reordered,
    ] {
        let out = tmp();
        let cfg = parse(&toml);
        pool(1)
            .install(|| run_from_checkpoint(&cfg, &ck, out.path()))
            .unwrap_or_else(|e| panic!("restart rejected: {e:?}\n{toml}"));
    }
}

/// "Restart determinism" (transform setting in the hash, both directions): an fft checkpoint does
/// not restart under matrix, and an fft checkpoint restarts under fft bitwise like an
/// uninterrupted fft run.
#[test]
fn fft_checkpoint_restarts_only_under_fft() {
    let fft = |n: usize| {
        parse(&set(
            &set(BASE, "sampling", "n_steps", &n.to_string()),
            "sampling",
            "normal_modes",
            "\"fft\"",
        ))
    };
    let (full, a, b, c) = (tmp(), tmp(), tmp(), tmp());
    pool(1).install(|| {
        run(&fft(40), full.path()).unwrap();
        run(&fft(20), a.path()).unwrap();
        let ck = a.path().join("checkpoint.bin");
        assert!(matches!(
            run_from_checkpoint(&with_steps(40), &ck, c.path()),
            Err(CliError::Checkpoint(_))
        ));
        run_from_checkpoint(&fft(40), &ck, b.path()).unwrap();
    });
    assert_eq!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&b.path().join("checkpoint.bin"))
    );
    let joined: Vec<String> = rows(a.path()).into_iter().chain(rows(b.path())).collect();
    assert_eq!(joined, rows(full.path()));
}

/// "Restart determinism": a run writes a checkpoint at its end even when n_steps is not a
/// multiple of checkpoint_every (13 steps, every 10), and restarting it reproduces the
/// uninterrupted run.
#[test]
fn checkpoint_written_at_end() {
    let (full, a, b) = (tmp(), tmp(), tmp());
    pool(1).install(|| {
        run(&with_steps(40), full.path()).unwrap();
        run(&with_steps(13), a.path()).unwrap();
        run_from_checkpoint(&with_steps(40), &a.path().join("checkpoint.bin"), b.path()).unwrap();
    });
    assert_eq!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&b.path().join("checkpoint.bin"))
    );
}

/// "Restart determinism": corrupt checkpoints are `CliError::Checkpoint`: empty, truncated by one
/// byte, halved, the first 8 bytes (magic) zeroed, and a file of unrelated bytes.
#[test]
fn corrupt_checkpoint_rejected() {
    let d = tmp();
    pool(1).install(|| run(&with_steps(10), d.path()).unwrap());
    let good = bytes(&d.path().join("checkpoint.bin"));
    let mut zeroed = good.clone();
    zeroed[..8].fill(0);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("truncated by one byte", good[..good.len() - 1].to_vec()),
        ("halved", good[..good.len() / 2].to_vec()),
        ("magic zeroed", zeroed),
        (
            "unrelated",
            b"not a checkpoint file at all, just text\n".repeat(50),
        ),
    ];
    let cfg = with_steps(40);
    for (what, data) in cases {
        let p = d.path().join("bad.bin");
        std::fs::write(&p, &data).unwrap();
        let out = tmp();
        match pool(1).install(|| run_from_checkpoint(&cfg, &p, out.path())) {
            Err(CliError::Checkpoint(_)) => {}
            other => panic!("{what}: expected Checkpoint error, got {other:?}"),
        }
    }
}
