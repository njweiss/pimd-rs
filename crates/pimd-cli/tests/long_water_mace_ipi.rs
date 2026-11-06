//! Periodic MACE-OFF23 water, in-process and chunked, vs i-PI with a Python MACE client
//! (MILESTONE_8 "M8a: Required tests" 8; SPEC P-6 as amended in M8a, I-1, T-1, T-2, E-2, E-3,
//! C-2, D-1). Features `torch` and `long-validation` (CHECK_LONG=1 CHECK_TORCH=1, release); needs
//! CUDA and `PIMD_MACE_MODEL` (the MACE-OFF23 small model file; skips with a message otherwise).
//!
//! Reference: `data/ipi_water_mace.json`, written by `data/gen_ipi_water_mace.py` (i-PI driving
//! one periodic Python MACE client, `clients/mace_pbc_client.py`: ASE `MACECalculator`, float32,
//! CUDA, deterministic; slow, CHECK_IPI only). The model file's FNV-1a must equal the generator's
//! record. The engine side exports the model (float64, `export_mace.py`, Q26) into a temporary
//! directory and runs `pimd_cli::run` with kind = "mace", device "cuda", dtype "float32" (the
//! adapter casts, D-1), `deterministic = true`, and `beads_per_graph = 3` (P = 8: chunks 3 + 3 + 2;
//! P = 1: c ≥ P, one graph), with positions and symbols from an xyz file (`positions_xyz`) written
//! from the same `m8_boxes.json` box `water_w64` (64 waters, L = 12.4295687235 Å).
//!
//! Same system and protocol on both sides (read from the JSON where it records them): O H H with
//! 15.999, 1.008, 1.008 amu, every bead at the box positions, 300 K, P = 1 and P = 8, dt = 0.5 fs,
//! PILE-L with τ_0 = 100 fs and γ_k = 2ω_k, BAOAB with the exact free ring-polymer propagator
//! (i-PI `splitting="baoab"`), equilibration and production lengths of the JSON. Trajectories
//! differ (each code draws its own momenta and noise; seeds differ); averages are compared.
//!
//! Compared, per molecule: ⟨E_vir⟩ and ⟨V̄⟩ (Eh; `observables.csv` per atom × 3) and g_OO(r)
//! per bin (bead-averaged O–O radial distribution, minimum image C-2, 60 bins of 0.1 Å on
//! [0, 6) Å, normalised by the ideal-gas pair count n_O(n_O − 1)/2 · shell volume / V; computed
//! here from `trajectory.xyz`, Å), each within 5σ_combined (WORKFLOW tolerance policy), with
//! σ = the largest of the block-averaged standard errors for blocks of 0.5, 1 and 2 ps (the
//! generator's procedure; no assertion on their ratio, see the generator). Bias allowance 0 (brief:
//! same splitting and dt on both sides, so the O(dt²) bias is common; no dt/2 run was made).
//!
//! Results with the throwaway reference implementation (2026-11-06, release, RTX 4060, both P
//! concurrently; per molecule): P = 1: ⟨E_vir⟩ = −76.490804 ± 0.000072 Eh vs i-PI −76.490630 ±
//! 0.000033 (−2.19 σ_c), ⟨V̄⟩ likewise (−2.19 σ_c), g_OO worst bin 2.75 σ_c of 60; P = 8:
//! ⟨E_vir⟩ = −76.476828 ± 0.000063 Eh vs −76.476847 ± 0.000073 (0.19 σ_c), ⟨V̄⟩ 0.19 σ_c, g_OO
//! worst bin 1.85 σ_c. The engine's block errors still grow from 0.5 to 2 ps blocks (P = 1: 4.1,
//! 5.3, 7.2e-5 Eh), hence σ = the largest. Engine run time 2 689 s (P = 1) and 6 394 s (P = 8).
#![cfg(all(feature = "torch", feature = "long-validation"))]

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use common::{observables, parse, read_trajectory, tmp};
use lammps::Json;
use pimd_cli::run;

const SEED: u64 = 80808;
const N_WATERS: usize = 64;
const BLOCK_STEPS: usize = 1000;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_json(path: &Path) -> Json {
    Json::parse(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
}

fn reference() -> &'static Json {
    static J: OnceLock<Json> = OnceLock::new();
    J.get_or_init(|| read_json(&root().join("crates/pimd-cli/tests/data/ipi_water_mace.json")))
}

fn string(j: &Json) -> String {
    match j {
        Json::Str(s) => s.clone(),
        other => panic!("expected a string, got {other:?}"),
    }
}

/// The `ml` environment's Python (direct when present, else through pixi).
fn ml_python() -> Command {
    let py = root().join(".pixi/envs/ml/bin/python");
    if py.exists() {
        Command::new(py)
    } else {
        let mut c = Command::new("pixi");
        c.args(["run", "-e", "ml", "python"]);
        c
    }
}

/// 64-bit FNV-1a (as the generator's).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The temporary directory with the float64 export and the xyz file, or `None` (with a message)
/// without `PIMD_MACE_MODEL` or without CUDA.
fn setup() -> Option<&'static (tempfile::TempDir, PathBuf, PathBuf)> {
    static S: OnceLock<Option<(tempfile::TempDir, PathBuf, PathBuf)>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(model) = std::env::var_os("PIMD_MACE_MODEL").map(PathBuf::from) else {
            eprintln!(
                "SKIP MACE-OFF23 water vs i-PI: set PIMD_MACE_MODEL to the MACE-OFF23 small model \
                 file (e.g. ff_models/mace_off23/MACE-OFF23_small.model)"
            );
            return None;
        };
        if !tch_cuda() {
            eprintln!("SKIP MACE-OFF23 water vs i-PI: libtorch sees no CUDA device");
            return None;
        }
        assert!(
            model.is_file(),
            "PIMD_MACE_MODEL={} is not a file",
            model.display()
        );
        let got = format!("{:#018x}", fnv1a64(&std::fs::read(&model).unwrap()));
        assert_eq!(
            got,
            string(reference().at(&["model", "fnv1a64"])),
            "PIMD_MACE_MODEL={} is not the model of ipi_water_mace.json",
            model.display()
        );
        let d = tmp();
        let export = d.path().join("export.pt");
        let st = ml_python()
            .current_dir(root())
            .arg(root().join("crates/pimd-torch/scripts/export_mace.py"))
            .arg(&model)
            .arg("--out")
            .arg(&export)
            .stdin(Stdio::null())
            .status()
            .expect("cannot run export_mace.py (pixi ml env)");
        assert!(st.success(), "export_mace.py failed");
        let xyz = d.path().join("water_w64.xyz");
        let (symbols, pos, _) = water_box();
        let mut s = format!("{}\nwater_w64 from m8_boxes.json\n", symbols.len());
        for (sym, x) in symbols.iter().zip(&pos) {
            s.push_str(&format!("{sym} {} {} {}\n", x[0], x[1], x[2]));
        }
        std::fs::write(&xyz, s).unwrap();
        Some((d, export, xyz))
    })
    .as_ref()
}

fn tch_cuda() -> bool {
    pimd_torch::MacePotential::load(
        &root().join("crates/pimd-torch/tests/data/tiny_mace_f64.pt"),
        &[8, 1, 1],
        pimd_torch::Device::Cuda(0),
        pimd_torch::Dtype::F32,
        1,
        true,
    )
    .is_ok()
}

/// (symbols, positions Å, L Å) of the box `water_w64`.
fn water_box() -> (Vec<String>, Vec<[f64; 3]>, f64) {
    let j = read_json(&root().join("crates/pimd-cli/tests/data/m8_boxes.json"));
    let b = j.at(&["boxes", "water_w64"]);
    let symbols: Vec<String> = b.get("symbols").arr().iter().map(string).collect();
    let pos = b
        .get("positions_angstrom")
        .arr()
        .iter()
        .map(|x| {
            let v = x.nums();
            [v[0], v[1], v[2]]
        })
        .collect();
    (symbols, pos, b.get("box_angstrom").num())
}

fn run_ref(p: usize) -> &'static Json {
    reference().at(&["runs", &format!("p{p}")])
}

/// The engine config for P beads (see the module doc).
fn config(p: usize, export: &Path, xyz: &Path) -> String {
    let (symbols, _, l) = water_box();
    let r = reference();
    let masses: Vec<String> = symbols
        .iter()
        .map(|s| format!("{}", r.at(&["system", "masses_amu", s.as_str()]).num()))
        .collect();
    let sys = r.get("system");
    let pr = r.get("protocol");
    let chunk = if p > 1 { "beads_per_graph = 3\n" } else { "" };
    format!(
        r#"
[system]
n_atoms = {n}
masses_amu = [{masses}]
box_angstrom = {l}
positions_xyz = "{xyz}"

[potential]
kind = "mace"
model = "{export}"
device = "cuda"
dtype = "float32"
deterministic = true
{chunk}
[sampling]
temperature_kelvin = {t}
n_beads = {p}
dt_fs = {dt}
tau0_fs = {tau}
n_equil = {ne}
n_steps = {np}
stride = {stride}
seed = {SEED}

[output]
checkpoint_every = {np}
trajectory_every = {ts}
"#,
        n = symbols.len(),
        masses = masses.join(", "),
        xyz = xyz.display(),
        export = export.display(),
        t = sys.get("temperature_kelvin").num(),
        dt = sys.get("dt_fs").num(),
        tau = sys.get("tau0_fs").num(),
        ne = run_ref(p).get("n_equil").num() as usize,
        np = run_ref(p).get("n_prod").num() as usize,
        stride = pr.get("stride").num() as usize,
        ts = pr.get("traj_stride").num() as usize,
    )
}

/// Mean and block standard errors with blocks of `len`, 2·`len` and 4·`len` samples (std of the
/// block means / √n_blocks, trailing partial block dropped); σ is the largest (generator's rule).
fn mean_sigma(x: &[f64], len: usize) -> (f64, [f64; 3]) {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let s = [1, 2, 4].map(|k| {
        let l = k * len;
        let nb = x.len() / l;
        assert!(nb >= 5, "too few blocks: {nb}");
        let means: Vec<f64> = (0..nb)
            .map(|b| x[b * l..(b + 1) * l].iter().sum::<f64>() / l as f64)
            .collect();
        let mm = means.iter().sum::<f64>() / nb as f64;
        let var = means.iter().map(|v| (v - mm).powi(2)).sum::<f64>() / (nb - 1) as f64;
        (var / nb as f64).sqrt()
    });
    (m, s)
}

/// Bead-averaged g_OO of one frame (rows: bead-major, Å), `bins` bins of `width` Å.
fn g_frame(
    rows: &[(String, [f64; 3])],
    p: usize,
    n: usize,
    l: f64,
    bins: usize,
    width: f64,
) -> Vec<f64> {
    let n_o = N_WATERS;
    let mut acc = vec![0.0; bins];
    for j in 0..p {
        let o: Vec<[f64; 3]> = (0..n_o).map(|m| rows[j * n + 3 * m].1).collect();
        for (m, r) in (0..n_o).map(|m| (m, &rows[j * n + 3 * m].0)) {
            assert_eq!(r, "O", "atom {} of bead {j} must be O", 3 * m);
        }
        for a in 0..n_o {
            for b in a + 1..n_o {
                let mut r2 = 0.0;
                for (xb, xa) in o[b].iter().zip(&o[a]) {
                    let mut d = xb - xa;
                    d -= l * (d / l).round();
                    r2 += d * d;
                }
                let k = (r2.sqrt() / width).floor() as usize;
                if k < bins {
                    acc[k] += 1.0;
                }
            }
        }
    }
    let pairs = (n_o * (n_o - 1)) as f64 / 2.0;
    (0..bins)
        .map(|k| {
            let (lo, hi) = (k as f64 * width, (k + 1) as f64 * width);
            let shell = 4.0 * std::f64::consts::PI / 3.0 * (hi.powi(3) - lo.powi(3));
            acc[k] / p as f64 / (pairs * shell / l.powi(3))
        })
        .collect()
}

fn check(what: &str, got: (f64, f64), want: (f64, f64), failures: &mut Vec<String>) -> f64 {
    let sc = (got.1 * got.1 + want.1 * want.1).sqrt();
    let z = if sc > 0.0 {
        (got.0 - want.0) / sc
    } else if got.0 == want.0 {
        0.0
    } else {
        f64::INFINITY
    };
    if z.abs() > 5.0 {
        failures.push(format!(
            "{what}: {} ± {} vs i-PI {} ± {} ({z:.2} sigma_c)",
            got.0, got.1, want.0, want.1
        ));
    }
    z
}

/// P-6, I-1, T-1, T-2, E-2, E-3 (MILESTONE_8 test 8): P beads, engine vs i-PI.
fn compare_with_ipi(p: usize) {
    let Some((_d, export, xyz)) = setup() else {
        return;
    };
    let r = run_ref(p);
    let t = config(p, export, xyz);
    let out = tmp();
    let start = std::time::Instant::now();
    run(&parse(&t), out.path()).unwrap_or_else(|e| panic!("P = {p}: run failed: {e:?}"));
    eprintln!("P = {p}: engine run {:.0} s", start.elapsed().as_secs_f64());
    let pr = reference().get("protocol");
    let stride = pr.get("stride").num() as usize;
    let len = BLOCK_STEPS / stride;
    assert_eq!(pr.get("block_steps").num() as usize, BLOCK_STEPS);
    let rows = observables(out.path());
    assert_eq!(rows.len(), r.get("n_prod").num() as usize / stride);
    let mut failures = Vec::new();
    for (key, col) in [("e_vir", 2), ("v", 3)] {
        let x: Vec<f64> = rows.iter().map(|row| row[col] * 3.0).collect();
        let (m, s) = mean_sigma(&x, len);
        let sigma = s.iter().copied().fold(0.0, f64::max);
        let w = r.get(key);
        let z = check(
            &format!("P = {p} {key}"),
            (m, sigma),
            (w.get("mean").num(), w.get("sigma").num()),
            &mut failures,
        );
        eprintln!(
            "P = {p} {key}: {m:.6} ± {sigma:.6} Eh/molecule (blocks {:.2e} {:.2e} {:.2e}) vs i-PI \
             {:.6} ± {:.6} ({z:.2} sigma_c)",
            s[0],
            s[1],
            s[2],
            w.get("mean").num(),
            w.get("sigma").num()
        );
    }
    let (_, _, l) = water_box();
    let n = 3 * N_WATERS;
    let g = r.get("g_oo");
    let edges = g.get("edges_angstrom").nums();
    let bins = edges.len() - 1;
    let width = pr.get("g_rmax_angstrom").num() / pr.get("g_bins").num();
    assert_eq!(bins, pr.get("g_bins").num() as usize);
    let frames = read_trajectory(&out.path().join("trajectory.xyz"), p, n);
    assert_eq!(frames.len(), g.get("n_frames").num() as usize);
    let per: Vec<Vec<f64>> = frames
        .iter()
        .map(|(_, rows)| g_frame(rows, p, n, l, bins, width))
        .collect();
    let tlen = g.get("block_frames").num() as usize;
    let (g_ref, s_ref) = (g.get("g").nums(), g.get("sigma").nums());
    let mut worst = (0.0f64, 0usize);
    for k in 0..bins {
        let col: Vec<f64> = per.iter().map(|f| f[k]).collect();
        let (m, s) = mean_sigma(&col, tlen);
        let sigma = s.iter().copied().fold(0.0, f64::max);
        let z = check(
            &format!(
                "P = {p} g_OO bin {k} [{:.1}, {:.1}) Å",
                edges[k],
                edges[k + 1]
            ),
            (m, sigma),
            (g_ref[k], s_ref[k]),
            &mut failures,
        );
        if z.abs() > worst.0 {
            worst = (z.abs(), k);
        }
    }
    let peak = (0..bins)
        .max_by(|&a, &b| g_ref[a].total_cmp(&g_ref[b]))
        .unwrap();
    eprintln!(
        "P = {p} g_OO: worst bin {} at {:.2} sigma_c; i-PI first peak bin {peak} g = {:.3}",
        worst.1, worst.0, g_ref[peak]
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// MILESTONE_8 test 8: the reference describes the same system and protocol as this test (box,
/// masses, temperature, dt, τ_0, block length, both P).
#[test]
fn same_system_as_ipi() {
    let r = reference();
    let (symbols, _, l) = water_box();
    assert_eq!(symbols.len(), 3 * N_WATERS);
    assert_eq!(string(r.at(&["system", "box"])), "water_w64");
    assert_eq!(r.at(&["system", "n_waters"]).num() as usize, N_WATERS);
    assert_eq!(r.at(&["system", "box_angstrom"]).num(), l);
    assert_eq!(r.at(&["system", "masses_amu", "O"]).num(), 15.999);
    assert_eq!(r.at(&["system", "masses_amu", "H"]).num(), 1.008);
    assert_eq!(r.at(&["system", "temperature_kelvin"]).num(), 300.0);
    assert_eq!(r.at(&["system", "dt_fs"]).num(), 0.5);
    assert_eq!(r.at(&["system", "tau0_fs"]).num(), 100.0);
    assert_eq!(
        r.at(&["protocol", "block_steps"]).num() as usize,
        BLOCK_STEPS
    );
    assert_eq!(string(r.at(&["protocol", "splitting"])), "baoab");
    for p in [1, 8] {
        assert!(run_ref(p).get("n_prod").num() >= 20.0 * BLOCK_STEPS as f64);
    }
}

/// P-6, I-1, T-1, T-2, E-2, E-3, D-1 (MILESTONE_8 test 8): classical (P = 1) MACE-OFF23 water, in
/// process (float32 CUDA) vs i-PI with a Python MACE client: ⟨E_vir⟩, ⟨V̄⟩ and g_OO within
/// 5σ_combined.
#[test]
fn water_p1_matches_ipi() {
    compare_with_ipi(1);
}

/// P-6, I-1, T-1, T-2, E-2, E-3, D-1 (MILESTONE_8 test 8): P = 8 MACE-OFF23 water, in process
/// (float32 CUDA, chunks of 3 + 3 + 2 beads) vs i-PI with a Python MACE client: ⟨E_vir⟩, ⟨V̄⟩ and
/// g_OO within 5σ_combined.
#[test]
fn water_p8_matches_ipi() {
    compare_with_ipi(8);
}
