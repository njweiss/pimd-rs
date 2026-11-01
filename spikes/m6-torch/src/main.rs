//! M6a spike: a TorchScript MACE model (export_mace.py) evaluated in-process through `tch` 0.26.
//! Throwaway record of the spike (MILESTONE_6 M6a); not maintained.
//!
//! ```text
//! m6-torch-spike info  MODEL
//! m6-torch-spike eval  MODEL GEOM OUT   [opts]   E/F per bead (eV, eV/Å) -> OUT
//! m6-torch-spike edges MODEL GEOM OUT   [opts]   Rust edge list -> OUT (format of ref_eval.py)
//! m6-torch-spike bench MODEL GEOM       [opts]   time per evaluation (all beads)
//! m6-torch-spike det   MODEL GEOM       [opts]   repeated calls bitwise equal? + fingerprint
//! opts: --device cpu|cuda  --dtype f64|f32 (casts the module if the export differs)
//!       --threads N  --mode batched|loop  --reps N  --warmup N  --edges-file F  --deterministic
//! ```
//! GEOM is the text format of make_geoms.py (Å). Everything here stays in eV and Å, the model's
//! units; the eV/Å <-> Eh/bohr conversion belongs to the M6b adapter (pimd_core::units, U-3).

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use tch::{CModule, Device, IValue, Kind, Tensor};

unsafe extern "C" {
    fn spike_set_deterministic(on: i32);
    fn spike_get_deterministic() -> i32;
}

/// Bead geometry: `pos` is `[P, N, 3]` in Å, `cell` the orthorhombic lengths or `None`.
struct Geom {
    n: usize,
    p: usize,
    cell: Option<[f64; 3]>,
    symbols: Vec<String>,
    pos: Vec<f64>,
}

fn read_geom(path: &str) -> Geom {
    let text = std::fs::read_to_string(path).expect("geometry file");
    let mut lines = text.lines();
    let mut field = |key: &str| {
        let l = lines.next().unwrap();
        let mut t = l.split_whitespace();
        assert_eq!(t.next(), Some(key), "expected {key}");
        t.map(str::to_string).collect::<Vec<_>>()
    };
    let n: usize = field("natoms")[0].parse().unwrap();
    let p: usize = field("nbeads")[0].parse().unwrap();
    let c: Vec<f64> = field("cell").iter().map(|x| x.parse().unwrap()).collect();
    let symbols = field("symbols");
    let pos: Vec<f64> = lines
        .take(n * p)
        .flat_map(|l| l.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>())
        .collect();
    assert_eq!(pos.len(), 3 * n * p);
    let cell = if c.iter().all(|&x| x == 0.0) { None } else { Some([c[0], c[1], c[2]]) };
    Geom { n, p, cell, symbols, pos }
}

/// Atomic number of an element symbol (the subset the spike needs).
fn z_of(s: &str) -> i64 {
    match s {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        _ => panic!("unknown symbol {s}"),
    }
}

/// Edge list over `nb` beads starting at bead `b0`, atoms numbered within the batch.
#[derive(Default)]
struct Edges {
    sender: Vec<i64>,
    receiver: Vec<i64>,
    shift: Vec<f64>,
    unit: Vec<f64>,
}

/// All ordered pairs i != j with |q_j - q_i + s| < r_max (strict), per bead. With a cell the
/// minimum image of SPEC C-2 (each component in [-L/2, L/2)) picks the single image, so
/// r_max < L_min/2 is required; shift s = -L * floor(d/L + 1/2), unit shift = s / L.
fn build_edges(g: &Geom, b0: usize, nb: usize, r_max: f64) -> Edges {
    if let Some(l) = g.cell {
        assert!(r_max < 0.5 * l.iter().cloned().fold(f64::INFINITY, f64::min), "r_max >= L/2");
    }
    let (n, r2max) = (g.n, r_max * r_max);
    let mut e = Edges::default();
    for b in b0..b0 + nb {
        let q = &g.pos[b * n * 3..(b + 1) * n * 3];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let (mut d, mut s, mut u) = ([0.0; 3], [0.0; 3], [0.0; 3]);
                for k in 0..3 {
                    d[k] = q[3 * j + k] - q[3 * i + k];
                    if let Some(l) = g.cell {
                        u[k] = -(d[k] / l[k] + 0.5).floor();
                        s[k] = u[k] * l[k];
                        d[k] += s[k];
                    }
                }
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < r2max {
                    e.sender.push(((b - b0) * n + i) as i64);
                    e.receiver.push(((b - b0) * n + j) as i64);
                    e.shift.extend_from_slice(&s);
                    e.unit.extend_from_slice(&u);
                }
            }
        }
    }
    e
}

/// Edges read from a ref_eval.py `--edges` file (mace's neighbour list), for beads b0 .. b0+nb.
fn file_edges(path: &str, n: usize, b0: usize, nb: usize) -> Edges {
    let mut e = Edges::default();
    for l in std::fs::read_to_string(path).unwrap().lines() {
        let t: Vec<&str> = l.split_whitespace().collect();
        let b: usize = t[1].parse().unwrap();
        if b < b0 || b >= b0 + nb {
            continue;
        }
        let off = ((b - b0) * n) as i64;
        e.sender.push(off + t[2].parse::<i64>().unwrap());
        e.receiver.push(off + t[3].parse::<i64>().unwrap());
        for k in 4..7 {
            e.shift.push(t[k].parse().unwrap());
        }
        for k in 7..10 {
            e.unit.push(t[k].parse().unwrap());
        }
    }
    e
}

struct Opts {
    device: Device,
    dtype: Option<Kind>,
    threads: Option<i32>,
    batched: bool,
    reps: usize,
    warmup: usize,
    edges_file: Option<String>,
    deterministic: bool,
}

fn parse_opts(args: &[String]) -> Opts {
    let mut o = Opts {
        device: Device::Cpu,
        dtype: None,
        threads: None,
        batched: true,
        reps: 20,
        warmup: 5,
        edges_file: None,
        deterministic: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().expect("option value").clone();
        match a.as_str() {
            "--device" => {
                o.device = match v().as_str() {
                    "cpu" => Device::Cpu,
                    "cuda" => Device::Cuda(0),
                    x => panic!("device {x}"),
                }
            }
            "--dtype" => {
                o.dtype = Some(match v().as_str() {
                    "f64" => Kind::Double,
                    "f32" => Kind::Float,
                    x => panic!("dtype {x}"),
                })
            }
            "--threads" => o.threads = Some(v().parse().unwrap()),
            "--mode" => o.batched = v() == "batched",
            "--reps" => o.reps = v().parse().unwrap(),
            "--warmup" => o.warmup = v().parse().unwrap(),
            "--edges-file" => o.edges_file = Some(v()),
            "--deterministic" => o.deterministic = true,
            x => panic!("unknown option {x}"),
        }
    }
    o
}

/// The loaded model and what the adapter needs to know about it.
struct Model {
    module: CModule,
    r_max: f64,
    species_index: HashMap<i64, i64>,
    kind: Kind,
    device: Device,
}

fn load(path: &str, o: &Opts) -> Model {
    let mut module = CModule::load_on_device(path, o.device).expect("load TorchScript model");
    module.set_eval();
    let code = match module.method_is::<IValue>("dtype_code", &[]).unwrap() {
        IValue::Int(c) => c,
        v => panic!("dtype_code: {v:?}"),
    };
    let export_kind = if code == 64 { Kind::Double } else { Kind::Float };
    let kind = o.dtype.unwrap_or(export_kind);
    if kind != export_kind {
        module.to(o.device, kind, false);
    }
    let r_max = module.method_ts::<Tensor>("r_max", &[]).unwrap().double_value(&[]);
    let z: Vec<i64> =
        Vec::try_from(module.method_ts::<Tensor>("atomic_numbers", &[]).unwrap().to(Device::Cpu))
            .unwrap();
    let species_index = z.iter().enumerate().map(|(k, &z)| (z, k as i64)).collect();
    Model { module, r_max, species_index, kind, device: o.device }
}

/// Energies `[nb]` (eV) and forces `[nb*N*3]` (eV/Å) of beads b0 .. b0+nb as one batched graph.
fn evaluate(m: &Model, g: &Geom, b0: usize, nb: usize, edges_file: Option<&str>) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    let e = match edges_file {
        Some(f) => file_edges(f, n, b0, nb),
        None => build_edges(g, b0, nb, m.r_max),
    };
    let ne = e.sender.len() as i64;
    let (dev, kind) = (m.device, m.kind);
    let pos = Tensor::from_slice(&g.pos[b0 * n * 3..(b0 + nb) * n * 3])
        .view([(nb * n) as i64, 3])
        .to_kind(kind)
        .to_device(dev);
    let sp: Vec<i64> = (0..nb)
        .flat_map(|_| g.symbols.iter().map(|s| m.species_index[&z_of(s)]))
        .collect();
    let species = Tensor::from_slice(&sp).to_device(dev);
    let mut ei = e.sender.clone();
    ei.extend_from_slice(&e.receiver);
    let edge_index = Tensor::from_slice(&ei).view([2, ne]).to_device(dev);
    let shifts = Tensor::from_slice(&e.shift).view([ne, 3]).to_kind(kind).to_device(dev);
    let unit = Tensor::from_slice(&e.unit).view([ne, 3]).to_kind(kind).to_device(dev);
    let ptr: Vec<i64> = (0..=nb).map(|b| (b * n) as i64).collect();
    let ptr = Tensor::from_slice(&ptr).to_device(dev);
    let cell = match g.cell {
        Some(l) => Tensor::from_slice(&[l[0], 0.0, 0.0, 0.0, l[1], 0.0, 0.0, 0.0, l[2]])
            .view([1, 3, 3])
            .repeat([nb as i64, 1, 1]),
        None => Tensor::zeros([nb as i64, 3, 3], (Kind::Double, Device::Cpu)),
    }
    .to_kind(kind)
    .to_device(dev);
    let out = m
        .module
        .forward_is(&[
            IValue::Tensor(pos),
            IValue::Tensor(species),
            IValue::Tensor(edge_index),
            IValue::Tensor(shifts),
            IValue::Tensor(ptr),
            IValue::Bool(false),
            IValue::Tensor(unit),
            IValue::Tensor(cell),
        ])
        .expect("forward");
    let IValue::Tuple(v) = out else { panic!("expected tuple") };
    let t = |x: &IValue| match x {
        IValue::Tensor(t) => Vec::<f64>::try_from(
            t.to_kind(Kind::Double).to_device(Device::Cpu).flatten(0, -1),
        )
        .unwrap(),
        _ => panic!("expected tensor"),
    };
    (t(&v[0]), t(&v[1]))
}

/// All beads, batched (one graph) or one call per bead.
fn evaluate_all(m: &Model, g: &Geom, o: &Opts) -> (Vec<f64>, Vec<f64>) {
    if o.batched {
        evaluate(m, g, 0, g.p, o.edges_file.as_deref())
    } else {
        let (mut e, mut f) = (Vec::new(), Vec::new());
        for b in 0..g.p {
            let (eb, fb) = evaluate(m, g, b, 1, o.edges_file.as_deref());
            e.extend(eb);
            f.extend(fb);
        }
        (e, f)
    }
}

fn setup(o: &Opts) {
    if let Some(t) = o.threads {
        tch::set_num_threads(t);
    }
    unsafe { spike_set_deterministic(i32::from(o.deterministic)) };
}

fn describe(m: &Model, g: &Geom, o: &Opts) -> String {
    format!(
        "device {:?} kind {:?} threads {} deterministic {} mode {} P {} N {}",
        m.device,
        m.kind,
        tch::get_num_threads(),
        unsafe { spike_get_deterministic() },
        if o.batched { "batched" } else { "loop" },
        g.p,
        g.n
    )
}

fn fingerprint(e: &[f64], f: &[f64]) -> u64 {
    // FNV-1a over the bit patterns: equal fingerprints <=> (almost surely) bitwise-equal results.
    let mut h: u64 = 0xcbf29ce484222325;
    for x in e.iter().chain(f) {
        for byte in x.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    match cmd {
        "info" => {
            let o = parse_opts(&args[3..]);
            setup(&o);
            let m = load(&args[2], &o);
            println!(
                "r_max {} A, species {:?}, kind {:?}, cuda {} ({} devices), threads {}, \
                 interop threads {}",
                m.r_max,
                m.species_index,
                m.kind,
                tch::Cuda::is_available(),
                tch::Cuda::device_count(),
                tch::get_num_threads(),
                tch::get_num_interop_threads()
            );
        }
        "eval" => {
            let o = parse_opts(&args[5..]);
            setup(&o);
            let m = load(&args[2], &o);
            let g = read_geom(&args[3]);
            let (e, f) = evaluate_all(&m, &g, &o);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&args[4]).unwrap());
            for b in 0..g.p {
                writeln!(w, "E {b} {:.17e}", e[b]).unwrap();
                for i in 0..g.n {
                    let x = &f[(b * g.n + i) * 3..(b * g.n + i + 1) * 3];
                    writeln!(w, "F {b} {i} {:.17e} {:.17e} {:.17e}", x[0], x[1], x[2]).unwrap();
                }
            }
            eprintln!("eval: {}", describe(&m, &g, &o));
        }
        "edges" => {
            let o = parse_opts(&args[5..]);
            let m = load(&args[2], &o);
            let g = read_geom(&args[3]);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&args[4]).unwrap());
            for b in 0..g.p {
                let e = build_edges(&g, b, 1, m.r_max);
                for k in 0..e.sender.len() {
                    let (s, u) = (&e.shift[3 * k..3 * k + 3], &e.unit[3 * k..3 * k + 3]);
                    writeln!(
                        w,
                        "edge {b} {} {} {:.17e} {:.17e} {:.17e} {} {} {}",
                        e.sender[k], e.receiver[k], s[0], s[1], s[2], u[0], u[1], u[2]
                    )
                    .unwrap();
                }
            }
        }
        "bench" => {
            let o = parse_opts(&args[4..]);
            setup(&o);
            let m = load(&args[2], &o);
            let g = read_geom(&args[3]);
            for _ in 0..o.warmup {
                evaluate_all(&m, &g, &o);
            }
            let mut t: Vec<f64> = (0..o.reps)
                .map(|_| {
                    let t0 = Instant::now();
                    evaluate_all(&m, &g, &o);
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            t.sort_by(f64::total_cmp);
            println!(
                "bench {} | median {:.3} ms  min {:.3} ms  max {:.3} ms  per bead {:.3} ms",
                describe(&m, &g, &o),
                t[t.len() / 2],
                t[0],
                t[t.len() - 1],
                t[t.len() / 2] / g.p as f64
            );
        }
        "det" => {
            let o = parse_opts(&args[4..]);
            setup(&o);
            let m = load(&args[2], &o);
            let g = read_geom(&args[3]);
            let (e0, f0) = evaluate_all(&m, &g, &o);
            let fp0 = fingerprint(&e0, &f0);
            let (mut n_diff, mut max_de, mut max_df) = (0, 0.0f64, 0.0f64);
            for _ in 1..o.reps {
                let (e, f) = evaluate_all(&m, &g, &o);
                if fingerprint(&e, &f) != fp0 {
                    n_diff += 1;
                }
                max_de = e.iter().zip(&e0).fold(max_de, |a, (x, y)| a.max((x - y).abs()));
                max_df = f.iter().zip(&f0).fold(max_df, |a, (x, y)| a.max((x - y).abs()));
            }
            println!(
                "det {} | {} calls, {} differ from the first, max|dE| {:.3e} eV, max|dF| {:.3e} \
                 eV/A, fingerprint {:016x}",
                describe(&m, &g, &o),
                o.reps,
                n_diff,
                max_de,
                max_df,
                fp0
            );
        }
        _ => eprintln!("usage: see the module doc comment in src/main.rs"),
    }
}
