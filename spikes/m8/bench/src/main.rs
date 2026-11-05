//! M8 spike bench (docs/experiments/m8-spike.md, Q1 and Q2): times the merged
//! `pimd_torch::MacePotential::evaluate` on a periodic box of P beads, optionally split into
//! consecutive bead chunks (one `evaluate` per chunk), and reads GPU memory from `nvidia-smi`.
//! Also times a copy of the merged all-pairs edge build (crates/pimd-torch/src/mace.rs
//! `Edges::build` / `bead_edges`, same algorithm and rayon parallelism) and a spike-local cell-list
//! build, and checks that both give the same edge list. Prints one JSON line.
//!
//! m8-bench --model X.pt --xyz box.xyz --p 16 --dtype f32 --device cuda [--threads 4]
//!          [--chunk 4] [--sigma 0.05] [--seed 8] [--max-iter 10] [--budget-s 60] [--warm 2]
//!          [--nondet] [--edges-only]

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use pimd_core::units::BOHR_IN_ANGSTROM;
use pimd_core::{Beads, Cell, Rng};
use pimd_potentials::Potential;
use pimd_torch::{Device, Dtype, MacePotential, atomic_number};
use rayon::prelude::*;

struct Args {
    model: PathBuf,
    xyz: PathBuf,
    p: usize,
    dtype: Dtype,
    device: Device,
    threads: usize,
    chunk: usize,
    sigma: f64,
    seed: u64,
    max_iter: usize,
    budget_s: f64,
    warm: usize,
    edges_only: bool,
    deterministic: bool,
}

fn parse() -> Args {
    let v: Vec<String> = std::env::args().collect();
    let get = |k: &str| v.iter().position(|a| a == k).map(|i| v[i + 1].clone());
    let p: usize = get("--p").map_or(1, |s| s.parse().unwrap());
    Args {
        model: get("--model").expect("--model").into(),
        xyz: get("--xyz").expect("--xyz").into(),
        p,
        dtype: match get("--dtype").as_deref() {
            Some("f32") => Dtype::F32,
            _ => Dtype::F64,
        },
        device: match get("--device").as_deref() {
            Some("cuda") => Device::Cuda(0),
            _ => Device::Cpu,
        },
        threads: get("--threads").map_or(1, |s| s.parse().unwrap()),
        chunk: get("--chunk").map_or(p, |s| s.parse().unwrap()),
        sigma: get("--sigma").map_or(0.05, |s| s.parse().unwrap()),
        seed: get("--seed").map_or(8, |s| s.parse().unwrap()),
        max_iter: get("--max-iter").map_or(10, |s| s.parse().unwrap()),
        budget_s: get("--budget-s").map_or(60.0, |s| s.parse().unwrap()),
        warm: get("--warm").map_or(2, |s| s.parse().unwrap()),
        edges_only: v.iter().any(|a| a == "--edges-only"),
        deterministic: !v.iter().any(|a| a == "--nondet"),
    }
}

/// Extended xyz: cubic `Lattice="L 0 0 0 L 0 0 0 L"`, then `symbol x y z` (Å).
fn read_xyz(path: &PathBuf) -> (f64, Vec<String>, Vec<[f64; 3]>) {
    let text = std::fs::read_to_string(path).expect("xyz");
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let header = lines.next().unwrap();
    let lat = header.split("Lattice=\"").nth(1).unwrap();
    let l: f64 = lat.split_whitespace().next().unwrap().parse().unwrap();
    let mut sym = Vec::new();
    let mut pos = Vec::new();
    for line in lines.take(n) {
        let w: Vec<&str> = line.split_whitespace().collect();
        sym.push(w[0].to_string());
        pos.push([1, 2, 3].map(|k| w[k].parse::<f64>().unwrap()));
    }
    (l, sym, pos)
}

/// GPU memory in use on device 0, MiB (all processes), or -1 if nvidia-smi fails.
fn gpu_mem() -> i64 {
    Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(-1)
}

/// Peak resident set of this process, MiB.
fn vm_hwm() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse::<f64>().ok()))
        })
        .map_or(-1.0, |kb| kb / 1024.0)
}

#[derive(Default)]
struct BeadEdges {
    sender: Vec<i64>,
    receiver: Vec<i64>,
    shift: Vec<f64>,
    unit: Vec<f64>,
}

#[derive(Default)]
struct Edges {
    beads: Vec<BeadEdges>,
    index: Vec<i64>,
    shift: Vec<f64>,
    unit: Vec<f64>,
}

impl Edges {
    /// Copy of the merged `Edges::build` (P-6), with the per-bead builder as a parameter.
    fn build(
        &mut self,
        q: &Beads,
        cell: &Cell,
        r_max: f64,
        f: fn(&Beads, usize, &Cell, f64, &mut BeadEdges),
    ) {
        let p = q.n_beads();
        self.beads.resize_with(p, BeadEdges::default);
        self.beads
            .par_iter_mut()
            .enumerate()
            .for_each(|(b, e)| f(q, b, cell, r_max, e));
        self.index.clear();
        self.shift.clear();
        self.unit.clear();
        for e in &self.beads {
            self.index.extend_from_slice(&e.sender);
        }
        for e in &self.beads {
            self.index.extend_from_slice(&e.receiver);
            self.shift.extend_from_slice(&e.shift);
            self.unit.extend_from_slice(&e.unit);
        }
    }
}

fn push(e: &mut BeadEdges, b: usize, n: usize, i: usize, j: usize, d: [f64; 3], v: [f64; 3], l: [f64; 3]) {
    let unit = [0, 1, 2].map(|k| ((v[k] - d[k]) / l[k]).round());
    e.sender.push((b * n + i) as i64);
    e.receiver.push((b * n + j) as i64);
    e.shift.extend((0..3).map(|k| unit[k] * l[k] * BOHR_IN_ANGSTROM));
    e.unit.extend_from_slice(&unit);
}

fn clear(e: &mut BeadEdges) {
    e.sender.clear();
    e.receiver.clear();
    e.shift.clear();
    e.unit.clear();
}

/// The merged all-pairs `bead_edges` (with a cell).
fn all_pairs(q: &Beads, b: usize, cell: &Cell, r_max: f64, e: &mut BeadEdges) {
    let n = q.n_atoms();
    let l = cell.lengths();
    let r2 = r_max * r_max;
    clear(e);
    for i in 0..n {
        let qi = q.get(b, i);
        for j in 0..n {
            if i == j {
                continue;
            }
            let qj = q.get(b, j);
            let d = [0, 1, 2].map(|k| qj[k] - qi[k]);
            let v = cell.min_image(d);
            if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < r2 {
                push(e, b, n, i, j, d, v, l);
            }
        }
    }
}

/// Spike-local cell list: atoms binned by wrapped position into cells of edge ≥ r_max; for each i
/// the candidates from the (deduplicated) neighbouring cells, sorted by j so the edge order equals
/// the all-pairs order; same distance test and shifts as `all_pairs`.
fn cell_list(q: &Beads, b: usize, cell: &Cell, r_max: f64, e: &mut BeadEdges) {
    let n = q.n_atoms();
    let l = cell.lengths();
    let r2 = r_max * r_max;
    clear(e);
    let nc = l.map(|x| ((x / r_max).floor() as usize).max(1));
    let idx = |r: [f64; 3]| -> [usize; 3] {
        let w = cell.wrap(r);
        [0, 1, 2].map(|k| ((w[k] / l[k] * nc[k] as f64) as usize).min(nc[k] - 1))
    };
    let flat = |c: [usize; 3]| (c[0] * nc[1] + c[1]) * nc[2] + c[2];
    let mut head = vec![Vec::<usize>::new(); nc[0] * nc[1] * nc[2]];
    let cells: Vec<[usize; 3]> = (0..n).map(|i| idx(q.get(b, i))).collect();
    for (i, c) in cells.iter().enumerate() {
        head[flat(*c)].push(i);
    }
    let mut cand = Vec::new();
    let mut nb = Vec::with_capacity(27);
    for i in 0..n {
        let c = cells[i];
        nb.clear();
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let m = [dx, dy, dz];
                    let cc = [0, 1, 2].map(|k| (c[k] as i64 + m[k]).rem_euclid(nc[k] as i64) as usize);
                    nb.push(flat(cc));
                }
            }
        }
        nb.sort_unstable();
        nb.dedup();
        cand.clear();
        for &f in &nb {
            cand.extend_from_slice(&head[f]);
        }
        cand.sort_unstable();
        let qi = q.get(b, i);
        for &j in &cand {
            if j == i {
                continue;
            }
            let qj = q.get(b, j);
            let d = [0, 1, 2].map(|k| qj[k] - qi[k]);
            let v = cell.min_image(d);
            if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < r2 {
                push(e, b, n, i, j, d, v, l);
            }
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = v.len();
    if k % 2 == 1 { v[k / 2] } else { 0.5 * (v[k / 2 - 1] + v[k / 2]) }
}

fn time_edges(e: &mut Edges, q: &Beads, cell: &Cell, r_max: f64, f: fn(&Beads, usize, &Cell, f64, &mut BeadEdges)) -> f64 {
    let mut t = Vec::new();
    let start = Instant::now();
    for _ in 0..10 {
        let s = Instant::now();
        e.build(q, cell, r_max, f);
        t.push(s.elapsed().as_secs_f64() * 1e3);
        if start.elapsed().as_secs_f64() > 20.0 {
            break;
        }
    }
    median(t)
}

fn main() {
    let a = parse();
    let (l_ang, sym, pos) = read_xyz(&a.xyz);
    let n = sym.len();
    let z: Vec<u8> = sym.iter().map(|s| atomic_number(s).expect("element")).collect();
    let cell = Cell::new([l_ang / BOHR_IN_ANGSTROM; 3]).unwrap();
    let mut rng = Rng::from_seed(a.seed);
    let mut data = Vec::with_capacity(a.p * n * 3);
    for _ in 0..a.p {
        for r in &pos {
            for &x in r {
                let dx = if a.p > 1 { a.sigma * rng.normal() } else { 0.0 };
                data.push((x + dx) / BOHR_IN_ANGSTROM);
            }
        }
    }
    let q = Beads::from_vec(a.p, n, data.clone()).unwrap();
    let chunks: Vec<Beads> = (0..a.p)
        .step_by(a.chunk)
        .map(|j0| {
            let j1 = (j0 + a.chunk).min(a.p);
            Beads::from_vec(j1 - j0, n, data[j0 * n * 3..j1 * n * 3].to_vec()).unwrap()
        })
        .collect();
    let dev = match a.device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => "cuda".to_string(),
    };
    let dt = match a.dtype {
        Dtype::F32 => "f32",
        Dtype::F64 => "f64",
    };
    let head = format!(
        "\"model\":\"{}\",\"xyz\":\"{}\",\"n_atoms\":{n},\"box_a\":{l_ang},\"p\":{},\"chunk\":{},\"dtype\":\"{dt}\",\"device\":\"{dev}\",\"threads\":{},\"deterministic\":{},\"seed\":{},\"sigma_a\":{}",
        a.model.display(),
        a.xyz.display(),
        a.p,
        a.chunk,
        a.threads,
        a.deterministic,
        a.seed,
        a.sigma
    );
    let mem0 = gpu_mem();
    let t_load = Instant::now();
    let mut pot = match MacePotential::load(&a.model, &z, a.device, a.dtype, a.threads, a.deterministic) {
        Ok(p) => p,
        Err(e) => {
            println!("{{{head},\"error\":\"load: {}\"}}", e.to_string().replace('"', "'"));
            return;
        }
    };
    let load_s = t_load.elapsed().as_secs_f64();
    let mem_load = gpu_mem();
    let r_max = pot.cutoff();

    // Edge builds (all-pairs copy of the merged code, cell list), timed and compared.
    let mut ea = Edges::default();
    let mut ec = Edges::default();
    let t_ap = time_edges(&mut ea, &q, &cell, r_max, all_pairs);
    let t_cl = time_edges(&mut ec, &q, &cell, r_max, cell_list);
    let same = ea.index == ec.index && ea.unit == ec.unit && ea.shift == ec.shift;
    let n_edges = ea.index.len() / 2;
    let edge_bytes = ea.index.len() * 8 + ea.shift.len() * 8 + ea.unit.len() * 8;
    let edge_part = format!(
        "\"r_max_bohr\":{r_max},\"edges\":{n_edges},\"edges_per_bead\":{},\"edge_host_mib\":{:.2},\"t_edges_allpairs_ms\":{t_ap:.3},\"t_edges_celllist_ms\":{t_cl:.3},\"celllist_equal\":{same},\"rayon_threads\":{}",
        n_edges / a.p,
        edge_bytes as f64 / 1048576.0,
        rayon::current_num_threads()
    );
    drop(ea);
    drop(ec);
    if a.edges_only {
        println!("{{{head},{edge_part}}}");
        return;
    }

    let eval = |pot: &mut MacePotential| -> Result<(f64, f64), String> {
        let mut e0 = f64::NAN;
        let mut fmax: f64 = 0.0;
        for (k, c) in chunks.iter().enumerate() {
            let ev = pot.evaluate(c, Some(&cell), None).map_err(|e| e.to_string())?;
            if k == 0 {
                e0 = ev.energies[0];
            }
            for f in ev.forces.as_slice() {
                fmax = fmax.max(f.abs());
            }
        }
        Ok((e0, fmax))
    };
    let mut times = Vec::new();
    let mut out = (f64::NAN, f64::NAN);
    let t_first = Instant::now();
    for _ in 0..a.warm {
        match eval(&mut pot) {
            Ok(r) => out = r,
            Err(e) => {
                let mem_err = gpu_mem();
                println!(
                    "{{{head},{edge_part},\"load_s\":{load_s:.2},\"mem_before_mib\":{mem0},\"mem_load_mib\":{mem_load},\"mem_after_mib\":{mem_err},\"error\":\"{}\"}}",
                    e.replace('"', "'").replace('\n', " ").chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect::<String>()
                );
                return;
            }
        }
    }
    let warm_s = t_first.elapsed().as_secs_f64();
    let start = Instant::now();
    while times.len() < a.max_iter {
        let s = Instant::now();
        out = eval(&mut pot).expect("evaluation failed after warm-up");
        times.push(s.elapsed().as_secs_f64() * 1e3);
        if times.len() >= 3 && start.elapsed().as_secs_f64() > a.budget_s {
            break;
        }
    }
    let mem_after = gpu_mem();
    let med = median(times.clone());
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "{{{head},{edge_part},\"load_s\":{load_s:.2},\"warm_s\":{warm_s:.2},\"n_timed\":{},\"ms_median\":{med:.3},\"ms_min\":{min:.3},\"ms_per_bead\":{:.4},\"us_per_edge\":{:.5},\"edge_share\":{:.4},\"mem_before_mib\":{mem0},\"mem_load_mib\":{mem_load},\"mem_after_mib\":{mem_after},\"graph_mem_mib\":{},\"vm_hwm_mib\":{:.0},\"e0_eh\":{},\"fmax_au\":{}}}",
        times.len(),
        med / a.p as f64,
        med * 1e3 / n_edges as f64,
        t_ap / med,
        mem_after - mem_load,
        vm_hwm(),
        out.0,
        out.1
    );
}
