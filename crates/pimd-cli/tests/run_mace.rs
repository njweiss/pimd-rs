//! Runs with `kind = "mace"` (MILESTONE_6 "Determinism" restart rows, "Config" hash coverage;
//! SPEC P-6, D-3 torch note, I-3/Q18). Feature `torch`; the tiny test model of `pimd-torch`.
//!
//! - The run drives `MacePotential` as configured: observables and trajectory equal a loop written
//!   in the test (`common::reference_run_with`) around `MacePotential::load` with the configured
//!   atomic numbers (O H H → 8 1 1), device, dtype and threads, to 1e-12 relative (the M4/M5
//!   reference-loop tolerance; same potential and integrator, so bitwise is expected).
//! - D-3: K + checkpoint + K equals 2K bitwise (final checkpoint) on CPU float64, and on CUDA with
//!   `deterministic = true` (skipped with a message without CUDA). With the TorchScript profiling
//!   executor left on, the first evaluation after the restart would round differently (spike Q6).
//! - Config hash (brief): covers the model file's bytes (not its path), the device kind (not the
//!   index), dtype, threads, deterministic and symbols. Checked as in M5 by restarting a checkpoint
//!   under a changed config: `CliError::Checkpoint` for covered fields; excluded ones restart. The
//!   "same model, other bytes" file is the export with a 4-byte ZIP archive comment added (the
//!   TorchScript archive stays loadable; checked with torch.jit.load during test authoring).
//!
//! libtorch threads are process-wide; every run here uses threads = 1 except the hash case
//! `threads = 2`, which must be refused before anything is loaded.
#![cfg(feature = "torch")]

mod common;
mod mace_common;

use std::path::{Path, PathBuf};

use common::{bytes, observables, parse, pool, reference_run_with, rel, set, tmp};
use mace_common::{mace_water_toml, tiny_export};
use pimd_cli::{CliError, Config, run, run_from_checkpoint};
use pimd_potentials::PotentialError;
use pimd_torch::{Device, Dtype, MacePotential};

const K: usize = 10;

fn cfg(model: &Path, n_steps: usize, extra: &[(&str, &str, &str)]) -> String {
    let mut t = mace_water_toml(model, n_steps);
    for (section, key, value) in extra {
        t = set(&t, section, key, value);
    }
    t
}

fn run_dir(toml: &str) -> tempfile::TempDir {
    let d = tmp();
    pool(1)
        .install(|| run(&parse(toml), d.path()))
        .unwrap_or_else(|e| panic!("run failed: {e:?}"));
    d
}

fn restart(toml: &str, ck: &Path) -> (Result<(), CliError>, tempfile::TempDir) {
    let out = tmp();
    let r = pool(1).install(|| run_from_checkpoint(&parse(toml), ck, out.path()).map(|_| ()));
    (r, out)
}

/// Whether the tiny model loads on CUDA device 0 (the CUDA rows are skipped otherwise).
fn cuda_ok() -> bool {
    match MacePotential::load(
        &tiny_export(),
        &[8, 1, 1],
        Device::Cuda(0),
        Dtype::F64,
        1,
        true,
    ) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("SKIP CUDA rows: cannot load on CUDA ({e})");
            false
        }
    }
}

/// The export with a 4-byte ZIP archive comment appended (end-of-central-directory comment
/// length 0 → 4): same model, different bytes.
fn other_bytes_copy(dir: &Path) -> PathBuf {
    let mut b = bytes(&tiny_export());
    let eocd = b
        .windows(4)
        .rposition(|w| w == b"PK\x05\x06")
        .expect("ZIP end of central directory");
    assert_eq!(
        &b[eocd + 20..eocd + 22],
        &[0, 0],
        "export already has a comment"
    );
    b[eocd + 20] = 4;
    b.extend_from_slice(b"pimd");
    let p = dir.join("tiny_commented.pt");
    std::fs::write(&p, b).unwrap();
    p
}

/// Observables and trajectory of `toml` against the test's loop around `pot`, 1e-12 relative.
fn compare_with_reference(toml: &str, pot: MacePotential, what: &str) -> Vec<Vec<f64>> {
    let c: Config = parse(toml);
    let d = run_dir(toml);
    let want = pool(1).install(|| reference_run_with(&c, Box::new(pot), Some(5)));
    let got = observables(d.path());
    assert_eq!(got.len(), want.rows.len(), "{what}: rows");
    assert!(!got.is_empty());
    for (g, w) in got.iter().zip(&want.rows) {
        assert_eq!(g[0] as usize, w.step);
        for (col, v) in [(1, w.e_prim), (2, w.e_vir), (3, w.v), (4, w.t_est)] {
            assert!(
                rel(g[col], v, 1e-300) <= 1e-12,
                "{what}: step {} column {col}: {} vs reference {v}",
                w.step,
                g[col]
            );
        }
    }
    let frames = common::read_trajectory(&d.path().join("trajectory.xyz"), 4, 3);
    assert_eq!(frames.len(), want.frames.len(), "{what}: frames");
    for ((s, rows), (s_ref, q)) in frames.iter().zip(&want.frames) {
        assert_eq!(s, s_ref);
        for j in 0..4 {
            for i in 0..3 {
                for k in 0..3 {
                    let x = q.get(j, i)[k] * 0.529177210903;
                    assert!(
                        rel(rows[j * 3 + i].1[k], x, 1e-300) <= 1e-12,
                        "{what}: frame {s}"
                    );
                }
            }
        }
    }
    got
}

/// P-6, I-1, D-1: a float64 run equals the test's loop around `MacePotential::load(model, [8, 1,
/// 1], Cpu, F64, 1, true)`; a float32 run equals the loop around the F32 potential and differs
/// from the float64 run (the dtype reaches the potential).
#[test]
fn mace_run_equals_reference_loop() {
    let ex = tiny_export();
    let t64 = cfg(&ex, 20, &[]);
    let pot = MacePotential::load(&ex, &[8, 1, 1], Device::Cpu, Dtype::F64, 1, true).unwrap();
    let r64 = compare_with_reference(&t64, pot, "float64");
    let t32 = cfg(&ex, 20, &[("potential", "dtype", "\"float32\"")]);
    let pot = MacePotential::load(&ex, &[8, 1, 1], Device::Cpu, Dtype::F32, 1, true).unwrap();
    let r32 = compare_with_reference(&t32, pot, "float32");
    let differs = r64
        .iter()
        .zip(&r32)
        .any(|(a, b)| a[1..4].iter().zip(&b[1..4]).any(|(x, y)| x != y));
    assert!(
        differs,
        "float32 run identical to float64 run: dtype ignored?"
    );
}

/// P-6: a symbol the model does not have (F; the tiny model knows H, C, N, O) parses but the run
/// fails with `CliError::Potential(InvalidParameter { what: "atomic number" })`.
#[test]
fn element_missing_from_model_fails_at_run() {
    let t = cfg(
        &tiny_export(),
        10,
        &[("system", "symbols", r#"["O", "H", "F"]"#)],
    );
    let d = tmp();
    match pool(1).install(|| run(&parse(&t), d.path())) {
        Err(CliError::Potential(PotentialError::InvalidParameter { what, .. })) => {
            assert_eq!(what, "atomic number")
        }
        other => panic!("expected Potential(InvalidParameter(atomic number)), got {other:?}"),
    }
}

/// K + checkpoint + K vs 2K with the given extra settings: final checkpoints byte-identical, and
/// the continued part's observables equal the last rows of the uninterrupted run.
fn restart_bitwise(extra: &[(&str, &str, &str)], what: &str) {
    let ex = tiny_export();
    let full = run_dir(&cfg(&ex, 2 * K, extra));
    let first = run_dir(&cfg(&ex, K, extra));
    let (r, cont) = restart(
        &cfg(&ex, 2 * K, extra),
        &first.path().join("checkpoint.bin"),
    );
    r.unwrap_or_else(|e| panic!("{what}: restart failed: {e:?}"));
    assert_eq!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&cont.path().join("checkpoint.bin")),
        "{what}: K + restart + K differs from 2K (profiling executor left on?)"
    );
    let all = observables(full.path());
    let tail = observables(cont.path());
    assert_eq!(
        &all[all.len() - tail.len()..],
        &tail[..],
        "{what}: observables"
    );
}

/// D-3 (torch note), I-3: restart is bitwise on CPU, float64 and float32, 1 thread.
#[test]
fn restart_bitwise_cpu() {
    restart_bitwise(&[], "cpu float64");
    restart_bitwise(&[("potential", "dtype", "\"float32\"")], "cpu float32");
}

/// D-3 (torch note): restart is bitwise on CUDA with `deterministic = true` (the default),
/// float64 and float32. Skipped with a message without CUDA.
#[test]
fn restart_bitwise_cuda_deterministic() {
    if !cuda_ok() {
        return;
    }
    restart_bitwise(&[("potential", "device", "\"cuda\"")], "cuda float64");
    restart_bitwise(
        &[
            ("potential", "device", "\"cuda\""),
            ("potential", "dtype", "\"float32\""),
        ],
        "cuda float32",
    );
}

/// D-3, config hash (brief): restarting a CPU float64 checkpoint under a config that changes the
/// model file's bytes, dtype, device kind (cpu → cuda), threads, deterministic or symbols is
/// `CliError::Checkpoint`; the same config, the same model under another path, and a removed
/// optional key at its default value restart, and the continuation equals the uninterrupted run.
#[test]
fn hash_covers_mace_settings() {
    let ex = tiny_export();
    let first = run_dir(&cfg(&ex, K, &[]));
    let ck = first.path().join("checkpoint.bin");
    let d = tmp();
    let commented = other_bytes_copy(d.path());
    let covered: Vec<(&str, String)> = vec![
        ("model bytes", cfg(&commented, 2 * K, &[])),
        (
            "dtype",
            cfg(&ex, 2 * K, &[("potential", "dtype", "\"float32\"")]),
        ),
        (
            "device kind",
            cfg(&ex, 2 * K, &[("potential", "device", "\"cuda\"")]),
        ),
        ("threads", cfg(&ex, 2 * K, &[("potential", "threads", "2")])),
        (
            "deterministic",
            cfg(&ex, 2 * K, &[("potential", "deterministic", "false")]),
        ),
        (
            "symbols",
            cfg(&ex, 2 * K, &[("system", "symbols", r#"["O", "H", "N"]"#)]),
        ),
    ];
    for (what, toml) in covered {
        match restart(&toml, &ck).0 {
            Err(CliError::Checkpoint(_)) => {}
            other => panic!("{what}: expected Checkpoint error, got {other:?}"),
        }
    }
    let full = run_dir(&cfg(&ex, 2 * K, &[]));
    let want = bytes(&full.path().join("checkpoint.bin"));
    let copy = d.path().join("renamed-copy.pt");
    std::fs::copy(&ex, &copy).unwrap();
    let explicit = cfg(
        &ex,
        2 * K,
        &[
            ("potential", "device", "\"cpu\""),
            ("potential", "dtype", "\"float64\""),
            ("potential", "threads", "1"),
            ("potential", "deterministic", "true"),
        ],
    );
    for (what, toml) in [
        ("same config", cfg(&ex, 2 * K, &[])),
        ("model under another path", cfg(&copy, 2 * K, &[])),
        ("defaults written out", explicit),
    ] {
        let (r, out) = restart(&toml, &ck);
        r.unwrap_or_else(|e| panic!("{what}: restart rejected: {e:?}"));
        assert_eq!(
            bytes(&out.path().join("checkpoint.bin")),
            want,
            "{what}: continuation differs from the uninterrupted run"
        );
    }
}

/// Config hash (brief): the device index is not covered: a "cuda:0" checkpoint restarts under
/// "cuda" (same device, bitwise equal to the uninterrupted run), and under "cuda:1" the restart is
/// not refused as a checkpoint mismatch (it runs, or fails loading the model when there is no
/// second device). Skipped with a message without CUDA.
#[test]
fn hash_excludes_device_index() {
    if !cuda_ok() {
        return;
    }
    let ex = tiny_export();
    let on = |dev: &str, n: usize| cfg(&ex, n, &[("potential", "device", &format!("\"{dev}\""))]);
    let first = run_dir(&on("cuda:0", K));
    let full = run_dir(&on("cuda:0", 2 * K));
    let ck = first.path().join("checkpoint.bin");
    let (r, out) = restart(&on("cuda", 2 * K), &ck);
    r.unwrap_or_else(|e| panic!("cuda:0 → cuda: restart rejected: {e:?}"));
    assert_eq!(
        bytes(&out.path().join("checkpoint.bin")),
        bytes(&full.path().join("checkpoint.bin"))
    );
    match restart(&on("cuda:1", 2 * K), &ck).0 {
        Ok(()) | Err(CliError::Potential(_)) => {}
        Err(e) => panic!("cuda:0 → cuda:1: the device index must not be hashed: {e:?}"),
    }
}
