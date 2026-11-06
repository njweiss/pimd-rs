//! Runs with `beads_per_graph` and `positions_xyz` (MILESTONE_8 "M8a: Required tests" 6; SPEC
//! P-6, D-3 chunk-size sentence, I-3). Feature `torch`; the tiny test model of `pimd-torch`, CPU,
//! float64, 1 libtorch thread.
//!
//! System: four water molecules in a 10.5 Å cubic box (hand-written below; molecules near faces,
//! so edges cross the boundary and H atoms lie outside [0, L)), P = 4, c = 3 (chunks of 3 + 1),
//! positions and symbols from an xyz file. The tiny model's r_max = 5 Å < L/2 = 5.25 Å.
//!
//! - D-3: the run is byte-identical across repeats (every output file) and to the run of the
//!   inline equivalent (`positions_angstrom` + `symbols`): MILESTONE_8 "Config" hash rule ("runs
//!   of the two are byte-identical").
//! - I-3, D-3: K + checkpoint + K equals 2K bitwise (final checkpoint bytes and observables).
//! - P-6: the run equals the test's own loop (`common::reference_run_with`) around
//!   `MacePotential::load(..).with_beads_per_graph(3)` to 1e-12 relative (the M6 run test's
//!   tolerance; bitwise is expected), and its trajectory and observables differ from the
//!   unchunked run's (c = 3 reaches the potential: a 4-bead graph and 3 + 1 bead graphs round
//!   differently, D-3; checked during test authoring for this system).
//! - run.log echoes the chunk size: the c = 3 and c = 2 logs differ outside the hash line, and
//!   the c = 3 log mentions it ("beads_per_graph", "beads per graph" or "chunk").
#![cfg(feature = "torch")]

mod common;
mod mace_common;

use std::path::Path;

use common::{bytes, observables, parse, pool, reference_run_with, rel, text, tmp};
use mace_common::tiny_export;
use pimd_cli::{run, run_from_checkpoint};
use pimd_torch::{Device, Dtype, MacePotential};

const K: usize = 10;
const L: f64 = 10.5;

/// Four waters, Å: O at a chosen centre, H at O + (0, ±0.7572, −0.4692) rotated by a quarter turn
/// about z for molecules 1 and 3; molecule 0 straddles the x = 0 and z = L faces, molecule 2 the
/// x = L face, molecule 3 the y = L face.
fn atoms() -> Vec<(&'static str, [f64; 3])> {
    let centres = [
        [0.31, 1.20, 10.38],
        [5.02, 5.47, 5.21],
        [10.21, 2.05, 4.03],
        [3.04, 10.12, 1.33],
    ];
    let mut out = Vec::new();
    for (m, c) in centres.iter().enumerate() {
        let h = |s: f64| {
            if m % 2 == 1 {
                [c[0] - s * 0.7572, c[1], c[2] - 0.4692]
            } else {
                [c[0], c[1] + s * 0.7572, c[2] - 0.4692]
            }
        };
        out.push(("O", *c));
        out.push(("H", h(1.0)));
        out.push(("H", h(-1.0)));
    }
    out
}

/// Coordinates rendered once, so the file and the inline config hold the same decimal strings.
fn fmt(x: f64) -> String {
    format!("{x:.6}")
}

fn write_xyz(dir: &Path) -> std::path::PathBuf {
    let mut s = format!("12\n4 waters, L = {L} A\n");
    for (sym, x) in atoms() {
        s.push_str(&format!(
            "{sym} {} {} {}\n",
            fmt(x[0]),
            fmt(x[1]),
            fmt(x[2])
        ));
    }
    let p = dir.join("waters.xyz");
    std::fs::write(&p, s).unwrap();
    p
}

/// The run config: 12 atoms, masses O 15.999 / H 1.008 amu, box 10.5 Å, kind = "mace" on the tiny
/// export with `beads_per_graph` = `c` (none if `None`), P = 4, 300 K, 0.5 fs, PILE-L τ_0 = 20 fs,
/// 4 equilibration steps, `n_steps` production steps, stride 2, trajectory every 5 steps,
/// checkpoint every 10. `system` holds either `positions_xyz` or the inline keys.
fn toml(system: &str, c: Option<usize>, n_steps: usize) -> String {
    let masses: Vec<&str> = atoms()
        .iter()
        .map(|(s, _)| if *s == "O" { "15.999" } else { "1.008" })
        .collect();
    let chunk = c.map_or(String::new(), |c| format!("beads_per_graph = {c}\n"));
    format!(
        r#"
[system]
n_atoms = 12
masses_amu = [{}]
box_angstrom = {L}
{system}

[potential]
kind = "mace"
model = "{}"
{chunk}
[sampling]
temperature_kelvin = 300.0
n_beads = 4
dt_fs = 0.5
tau0_fs = 20.0
n_equil = 4
n_steps = {n_steps}
stride = 2
seed = 8080

[output]
checkpoint_every = 10
trajectory_every = 5
"#,
        masses.join(", "),
        tiny_export().display()
    )
}

fn xyz_system(path: &Path) -> String {
    format!("positions_xyz = \"{}\"", path.display())
}

fn inline_system() -> String {
    let pos: Vec<String> = atoms()
        .iter()
        .map(|(_, x)| format!("[{}, {}, {}]", fmt(x[0]), fmt(x[1]), fmt(x[2])))
        .collect();
    let sym: Vec<String> = atoms().iter().map(|(s, _)| format!("\"{s}\"")).collect();
    format!(
        "symbols = [{}]\npositions_angstrom = [{}]",
        sym.join(", "),
        pos.join(", ")
    )
}

fn run_dir(t: &str) -> tempfile::TempDir {
    let d = tmp();
    pool(1)
        .install(|| run(&parse(t), d.path()))
        .unwrap_or_else(|e| panic!("run failed: {e:?}"));
    d
}

const FILES: [&str; 4] = [
    "observables.csv",
    "trajectory.xyz",
    "checkpoint.bin",
    "run.log",
];

/// D-3, MILESTONE_8 test 6: the c = 3 xyz run is byte-identical across repeats and to the inline
/// equivalent, in every output file.
#[test]
fn xyz_chunked_run_is_byte_identical_and_equals_inline() {
    let d = tmp();
    let x = write_xyz(d.path());
    let a = run_dir(&toml(&xyz_system(&x), Some(3), 2 * K));
    let b = run_dir(&toml(&xyz_system(&x), Some(3), 2 * K));
    let i = run_dir(&toml(&inline_system(), Some(3), 2 * K));
    for f in FILES {
        let fa = bytes(&a.path().join(f));
        assert!(!fa.is_empty(), "{f} empty");
        assert_eq!(fa, bytes(&b.path().join(f)), "{f}: repeat differs");
        assert_eq!(fa, bytes(&i.path().join(f)), "{f}: inline run differs");
    }
}

/// I-3, D-3, MILESTONE_8 test 6: K + checkpoint + K equals 2K bitwise with c = 3 (positions from
/// the xyz file): final checkpoints byte-identical, the continued observables equal the last rows
/// of the uninterrupted run.
#[test]
fn chunked_restart_is_bitwise() {
    let d = tmp();
    let x = write_xyz(d.path());
    let full = run_dir(&toml(&xyz_system(&x), Some(3), 2 * K));
    let first = run_dir(&toml(&xyz_system(&x), Some(3), K));
    let cont = tmp();
    pool(1)
        .install(|| {
            run_from_checkpoint(
                &parse(&toml(&xyz_system(&x), Some(3), 2 * K)),
                &first.path().join("checkpoint.bin"),
                cont.path(),
            )
        })
        .unwrap_or_else(|e| panic!("restart failed: {e:?}"));
    assert_eq!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&cont.path().join("checkpoint.bin")),
        "K + restart + K differs from 2K"
    );
    let all = observables(full.path());
    let tail = observables(cont.path());
    assert!(!tail.is_empty());
    assert_eq!(&all[all.len() - tail.len()..], &tail[..]);
}

/// P-6, MILESTONE_8 test 6: the c = 3 run equals the test's loop around the chunked potential
/// (observables and trajectory to 1e-12 relative), and differs from the unchunked run (c = 3
/// reaches the potential).
#[test]
fn chunked_run_uses_the_chunk_size() {
    let d = tmp();
    let x = write_xyz(d.path());
    let t = toml(&xyz_system(&x), Some(3), 2 * K);
    let cfg = parse(&t);
    let got = run_dir(&t);
    let pot = MacePotential::load(
        &tiny_export(),
        &[8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1],
        Device::Cpu,
        Dtype::F64,
        1,
        true,
    )
    .unwrap()
    .with_beads_per_graph(3)
    .unwrap();
    let want = pool(1).install(|| reference_run_with(&cfg, Box::new(pot), Some(5)));
    let rows = observables(got.path());
    assert_eq!(rows.len(), want.rows.len());
    for (g, w) in rows.iter().zip(&want.rows) {
        assert_eq!(g[0] as usize, w.step);
        for (col, v) in [(1, w.e_prim), (2, w.e_vir), (3, w.v), (4, w.t_est)] {
            assert!(
                rel(g[col], v, 1e-300) <= 1e-12,
                "step {} column {col}: {} vs reference {v}",
                w.step,
                g[col]
            );
        }
    }
    let frames = common::read_trajectory(&got.path().join("trajectory.xyz"), 4, 12);
    assert_eq!(frames.len(), want.frames.len());
    for ((s, rows), (s_ref, q)) in frames.iter().zip(&want.frames) {
        assert_eq!(s, s_ref);
        for j in 0..4 {
            for i in 0..12 {
                for k in 0..3 {
                    let xa = q.get(j, i)[k] * 0.529177210903;
                    assert!(rel(rows[j * 12 + i].1[k], xa, 1e-300) <= 1e-12, "frame {s}");
                }
            }
        }
    }
    // The checkpoints differ anyway (c = 3 < P enters the config hash), so compare the trajectory
    // and the observables.
    let plain = run_dir(&toml(&xyz_system(&x), None, 2 * K));
    let same = ["trajectory.xyz", "observables.csv"]
        .iter()
        .all(|f| bytes(&got.path().join(f)) == bytes(&plain.path().join(f)));
    assert!(
        !same,
        "c = 3 run identical to the unchunked run: beads_per_graph not passed to the potential?"
    );
}

/// MILESTONE_8 "Config": run.log echoes the chunk size when set: the c = 3 and c = 2 logs differ
/// outside the `config hash` line, and the c = 3 log mentions it.
#[test]
fn run_log_echoes_chunk_size() {
    let d = tmp();
    let x = write_xyz(d.path());
    let strip = |dir: &Path| -> String {
        text(&dir.join("run.log"))
            .lines()
            .filter(|l| !l.starts_with("config hash = "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let a = run_dir(&toml(&xyz_system(&x), Some(3), K));
    let b = run_dir(&toml(&xyz_system(&x), Some(2), K));
    assert_ne!(strip(a.path()), strip(b.path()), "chunk size not echoed");
    let log = text(&a.path().join("run.log")).to_lowercase();
    assert!(
        log.contains("beads_per_graph") || log.contains("beads per graph") || log.contains("chunk"),
        "run.log does not mention the chunk size:\n{log}"
    );
}
