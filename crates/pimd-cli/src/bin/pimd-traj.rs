//! `pimd-traj <trajectory.xyz>... --out <file.extxyz> [--beads] [--box-angstrom L] [--every n]
//! [--no-wrap] [--bond-angstrom 1.6]` (MILESTONE_9 M9b): converts `trajectory.xyz` files
//! (MILESTONE_5 "Trajectory output"), given in run order, to extended XYZ for OVITO, VMD and ASE.
//!
//! The files are streamed one frame at a time. Restart joining: a frame whose step is ≥ the next
//! file's first step is dropped (the rest of that file is not read). Molecules and their
//! make-whole offsets come from the first kept frame (QUESTIONS.md Q40 item 3); every frame then
//! moves each atom's beads and centroid by its offset plus one lattice vector per molecule.
//! All lengths are Å.

#![allow(clippy::needless_range_loop)]

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

/// Trajectory export for viewers: centroids (default) or all beads, molecules wrapped.
#[derive(Parser)]
#[command(name = "pimd-traj", version)]
struct Args {
    /// `trajectory.xyz` files in run order (restart parts first).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output extended-XYZ file.
    #[arg(long)]
    out: PathBuf,
    /// Write all P·N beads (bead-major, with a `bead` column) instead of the N centroids.
    #[arg(long)]
    beads: bool,
    /// Cubic box edge L in Å; enables `Lattice`/`pbc` and molecule wrapping.
    #[arg(long = "box-angstrom")]
    box_angstrom: Option<f64>,
    /// Keep every n-th frame.
    #[arg(long, default_value_t = 1)]
    every: usize,
    /// With a box, write positions unwrapped.
    #[arg(long = "no-wrap")]
    no_wrap: bool,
    /// Bond cutoff in Å for finding molecules (minimum image).
    #[arg(long = "bond-angstrom", default_value_t = 1.6)]
    bond_angstrom: f64,
}

fn main() -> ExitCode {
    match convert(&Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `<path>:<line>: <msg>`, line 1-based within the file (Q40 item 6).
fn at(path: &Path, line: usize, msg: impl std::fmt::Display) -> String {
    format!("{}:{line}: {msg}", path.display())
}

/// Line reader over one input file that counts 1-based line numbers.
struct Reader<'a> {
    path: &'a Path,
    inner: BufReader<File>,
    buf: String,
    line: usize,
}

impl<'a> Reader<'a> {
    fn open(path: &'a Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            path,
            inner: BufReader::new(f),
            buf: String::new(),
            line: 0,
        })
    }

    /// The next line's 1-based number, its text without the line ending, and whether it is the
    /// file's last line; `None` at end of file.
    fn next_line(&mut self) -> Result<Option<(usize, &str, bool)>, String> {
        self.buf.clear();
        let read = self
            .inner
            .read_line(&mut self.buf)
            .map_err(|e| at(self.path, self.line + 1, e))?;
        if read == 0 {
            return Ok(None);
        }
        self.line += 1;
        let end = self
            .inner
            .fill_buf()
            .map_err(|e| at(self.path, self.line + 1, e))?
            .is_empty();
        Ok(Some((
            self.line,
            self.buf.trim_end_matches(['\n', '\r']),
            end,
        )))
    }
}

/// Frame header `step=<s> beads=<P> atoms=<N> units=angstrom` → (step, P, N).
fn parse_header(line: &str) -> Result<(u64, usize, usize), String> {
    let (mut step, mut beads, mut atoms, mut units) = (None, None, None, None);
    for tok in line.split_whitespace() {
        let Some((key, value)) = tok.split_once('=') else {
            return Err(format!("malformed header field {tok:?}"));
        };
        match key {
            "step" => step = Some(value),
            "beads" => beads = Some(value),
            "atoms" => atoms = Some(value),
            "units" => units = Some(value),
            _ => {}
        }
    }
    let num = |name: &str, v: Option<&str>| -> Result<usize, String> {
        let v = v.ok_or_else(|| format!("header lacks {name}="))?;
        match v.parse::<usize>() {
            Ok(x) if x > 0 => Ok(x),
            _ => Err(format!("header {name}={v} is not a positive integer")),
        }
    };
    let s = step.ok_or("header lacks step=")?;
    let step = s
        .parse::<u64>()
        .map_err(|_| format!("header step={s} is not an integer"))?;
    let p = num("beads", beads)?;
    let n = num("atoms", atoms)?;
    if p.checked_mul(n).is_none() {
        return Err(format!("header beads={p} atoms={n} is too large"));
    }
    match units {
        Some("angstrom") => Ok((step, p, n)),
        Some(u) => Err(format!("header units={u}, expected angstrom")),
        None => Err("header lacks units=".to_string()),
    }
}

/// The first step of the first of `paths` that has a complete first header (a header line
/// followed by more lines; files without one hold no frames, F2), if that header is readable.
/// Errors are left for the full read of that file, so they are reported in reading order.
fn next_first_step(paths: &[PathBuf]) -> Option<u64> {
    for path in paths {
        let mut r = Reader::open(path).ok()?;
        if r.next_line().ok()?.is_none() {
            continue;
        }
        match r.next_line().ok()? {
            Some((_, header, false)) => return parse_header(header).ok().map(|h| h.0),
            _ => continue,
        }
    }
    None
}

/// Minimum-image displacement component (C-2): the result lies in [−L/2, L/2).
fn min_image(d: f64, l: f64) -> f64 {
    let half = 0.5 * l;
    let mut x = d - l * (d / l).round();
    if x >= half {
        x -= l;
    } else if x < -half {
        x += l;
    }
    x
}

/// Molecules and make-whole offsets from one frame's centroids `c` (Q40 item 3): connected
/// components with edges at C-2 distance strictly below `bond`, walked breadth-first (FIFO) from
/// each lowest-index unvisited atom, bonded neighbours in ascending index; atom j reached from i
/// gets offset n_i + k·L, k = −floor((c_j − c_i)/L + 1/2). Returns (root atom per atom, offset
/// per atom), Å.
fn molecules(c: &[[f64; 3]], l: f64, bond: f64) -> (Vec<usize>, Vec<[f64; 3]>) {
    let n = c.len();
    let mut root = vec![usize::MAX; n];
    let mut off = vec![[0.0; 3]; n];
    let mut queue = std::collections::VecDeque::new();
    for s in 0..n {
        if root[s] != usize::MAX {
            continue;
        }
        root[s] = s;
        queue.push_back(s);
        while let Some(i) = queue.pop_front() {
            for j in 0..n {
                if root[j] != usize::MAX {
                    continue;
                }
                let d = [0, 1, 2].map(|k| c[j][k] - c[i][k]);
                let r2: f64 = d.iter().map(|&x| min_image(x, l).powi(2)).sum();
                if r2.sqrt() < bond {
                    root[j] = s;
                    for k in 0..3 {
                        off[j][k] = off[i][k] - (d[k] / l + 0.5).floor() * l;
                    }
                    queue.push_back(j);
                }
            }
        }
    }
    (root, off)
}

/// One input frame: symbols and positions of the P·N rows, bead-major, Å.
struct Frame {
    step: u64,
    p: usize,
    n: usize,
    syms: Vec<String>,
    pos: Vec<[f64; 3]>,
}

/// Atom line `symbol x y z` with finite coordinates (F6).
fn parse_row(line: &str) -> Option<(&str, [f64; 3])> {
    let mut t = line.split_whitespace();
    let sym = t.next()?;
    let mut x = [0.0; 3];
    for v in &mut x {
        *v = t.next()?.parse().ok().filter(|v: &f64| v.is_finite())?;
    }
    t.next().is_none().then_some((sym, x))
}

/// Reads the P·N atom lines of a frame into `f`. `Ok(false)` at a premature end of file, or, with
/// `cut_ok` (a file other than the last, F1), when the file's last line is malformed (a cut line).
fn read_rows(r: &mut Reader, f: &mut Frame, cut_ok: bool) -> Result<bool, String> {
    let rows = f.p * f.n;
    f.pos.clear();
    for k in 0..rows {
        let path = r.path;
        let Some((lineno, line, end)) = r.next_line()? else {
            return Ok(false);
        };
        let Some((sym, x)) = parse_row(line) else {
            if cut_ok && end {
                return Ok(false);
            }
            return Err(at(path, lineno, format!("malformed atom line {line:?}")));
        };
        // Symbol strings are reused from frame to frame.
        match f.syms.get_mut(k) {
            Some(s) => {
                s.clear();
                s.push_str(sym);
            }
            None => f.syms.push(sym.to_string()),
        }
        f.pos.push(x);
    }
    Ok(true)
}

/// Checks the options, then writes the output; on an error after the output file was created,
/// removes it (F4).
fn convert(a: &Args) -> Result<(), String> {
    if a.every == 0 {
        return Err("--every must be at least 1".into());
    }
    if let Some(l) = a.box_angstrom
        && !(l.is_finite() && l > 0.0)
    {
        return Err(format!(
            "--box-angstrom must be positive and finite, got {l}"
        ));
    }
    if !(a.bond_angstrom.is_finite() && a.bond_angstrom > 0.0) {
        return Err(format!(
            "--bond-angstrom must be positive and finite, got {}",
            a.bond_angstrom
        ));
    }
    // F3: an output that does not exist yet cannot be an input.
    if let Ok(out) = a.out.canonicalize()
        && let Some(p) = a
            .inputs
            .iter()
            .find(|p| p.canonicalize().is_ok_and(|c| c == out))
    {
        return Err(format!(
            "--out {} is the input {}",
            a.out.display(),
            p.display()
        ));
    }
    let file = File::create(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let result = write_all(a, BufWriter::new(file));
    if result.is_err() {
        let _ = std::fs::remove_file(&a.out);
    }
    result
}

/// Streams the inputs into `w` (restart joining, `--every`, wrapping).
fn write_all(a: &Args, mut w: BufWriter<File>) -> Result<(), String> {
    let wrap_box = a.box_angstrom.filter(|_| !a.no_wrap);
    let out_err = |e: std::io::Error| format!("{}: {e}", a.out.display());

    let mut frame = Frame {
        step: 0,
        p: 0,
        n: 0,
        syms: Vec::new(),
        pos: Vec::new(),
    };
    let mut pn: Option<(usize, usize)> = None;
    let mut mols: Option<(Vec<usize>, Vec<[f64; 3]>)> = None;
    let mut index = 0usize;
    let mut cent: Vec<[f64; 3]> = Vec::new();
    let mut shift: Vec<[f64; 3]> = Vec::new();

    for (fi, path) in a.inputs.iter().enumerate() {
        let last = fi + 1 == a.inputs.len();
        let limit = next_first_step(&a.inputs[fi + 1..]);
        let mut r = Reader::open(path)?;
        let mut prev_step: Option<u64> = None;
        while let Some((count_line, count, _)) = r.next_line()? {
            let count: Option<usize> = count.trim().parse().ok();
            let partial = || at(path, count_line, "incomplete last frame");
            let first_in_file = prev_step.is_none();
            let Some((header_line, header, end)) = r.next_line()? else {
                if last && !first_in_file {
                    return Err(partial());
                }
                break;
            };
            // A frame cut at or inside its header line: dropped in a file other than the last
            // (F1); a file without a complete first header holds no frames (F2).
            if end && (first_in_file || !last) {
                break;
            }
            let count = count.ok_or_else(|| at(path, count_line, "malformed count line"))?;
            let (step, p, n) = parse_header(header).map_err(|e| at(path, header_line, e))?;
            if count != p * n {
                return Err(at(
                    path,
                    count_line,
                    format!("count {count} ≠ P·N = {}", p * n),
                ));
            }
            match pn {
                None => pn = Some((p, n)),
                Some((p0, n0)) if (p0, n0) != (p, n) => {
                    return Err(at(
                        path,
                        header_line,
                        format!(
                            "P = {p}, N = {n} differ from the first frame's P = {p0}, N = {n0}"
                        ),
                    ));
                }
                Some(_) => {}
            }
            if prev_step.is_some_and(|s| step <= s) {
                return Err(at(
                    path,
                    header_line,
                    format!("step {step} does not increase"),
                ));
            }
            prev_step = Some(step);
            if limit.is_some_and(|s| step >= s) {
                break;
            }
            (frame.step, frame.p, frame.n) = (step, p, n);
            if !read_rows(&mut r, &mut frame, !last)? {
                if last {
                    return Err(partial());
                }
                break;
            }
            if index.is_multiple_of(a.every) {
                write_frame(
                    &mut w, a, wrap_box, &frame, &mut mols, &mut cent, &mut shift,
                )
                .map_err(out_err)?;
            }
            index += 1;
        }
    }
    w.flush().map_err(out_err)
}

/// Writes one kept frame: centroids (or all beads), each atom moved by its make-whole offset plus
/// its molecule's wrap shift −floor(c/L)·L, c the offset centroid of the molecule's root (Q40
/// items 1–3). `mols` is filled from the first kept frame.
#[allow(clippy::too_many_arguments)]
fn write_frame(
    w: &mut impl Write,
    a: &Args,
    wrap_box: Option<f64>,
    f: &Frame,
    mols: &mut Option<(Vec<usize>, Vec<[f64; 3]>)>,
    cent: &mut Vec<[f64; 3]>,
    shift: &mut Vec<[f64; 3]>,
) -> std::io::Result<()> {
    let (p, n) = (f.p, f.n);
    cent.clear();
    for i in 0..n {
        let mut c = [0.0; 3];
        for j in 0..p {
            for k in 0..3 {
                c[k] += f.pos[j * n + i][k];
            }
        }
        cent.push(c.map(|v| v / p as f64));
    }
    shift.clear();
    match wrap_box {
        Some(l) => {
            let (root, off) = mols.get_or_insert_with(|| molecules(cent, l, a.bond_angstrom));
            for i in 0..n {
                let r = root[i];
                let wrap = [0, 1, 2].map(|k| -((cent[r][k] + off[r][k]) / l).floor() * l);
                shift.push([0, 1, 2].map(|k| off[i][k] + wrap[k]));
            }
        }
        None => shift.resize(n, [0.0; 3]),
    }

    writeln!(w, "{}", if a.beads { p * n } else { n })?;
    write!(w, "step={}", f.step)?;
    if let Some(l) = a.box_angstrom {
        write!(w, " Lattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\"")?;
    }
    if a.beads {
        writeln!(w, " Properties=species:S:1:pos:R:3:bead:I:1")?;
        for j in 0..p {
            for i in 0..n {
                let (x, s) = (f.pos[j * n + i], shift[i]);
                writeln!(
                    w,
                    "{} {} {} {} {j}",
                    f.syms[j * n + i],
                    x[0] + s[0],
                    x[1] + s[1],
                    x[2] + s[2]
                )?;
            }
        }
    } else {
        writeln!(w, " Properties=species:S:1:pos:R:3")?;
        for i in 0..n {
            let (c, s) = (cent[i], shift[i]);
            writeln!(
                w,
                "{} {} {} {}",
                f.syms[i],
                c[0] + s[0],
                c[1] + s[1],
                c[2] + s[2]
            )?;
        }
    }
    Ok(())
}
