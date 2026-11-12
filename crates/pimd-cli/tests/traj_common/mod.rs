//! Shared helpers for the `pimd-traj` tests (MILESTONE_9 M9b): the hand-written input and
//! expected outputs of required test 1, a runner for the binary, a small extended-XYZ reader and
//! the error-message checks (QUESTIONS.md Q40).
//!
//! Every coordinate in the hand-written data is a multiple of 0.25 Å (inputs) and the box is
//! L = 10 Å, so centroids (sums of two such values, halved), and lattice shifts k·L, are exact in
//! f64 whatever the summation order or the way the shift is computed (−floor(c/L)·L, `rem_euclid`,
//! …). The expected text is therefore exact, written with `{}` (shortest round trip).
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Test 1 input: `trajectory.xyz` (MILESTONE_5 "Trajectory output") with P = 2, N = 4, three
/// frames (steps 10, 20, 30), bead-major, Å, unwrapped, numbers as the engine writes them (`{}`).
/// Molecule A = atoms 0 (O) and 1 (H); molecule B = atoms 2 (H) and 3 (F).
///
/// Centroids (mean of the two beads; hand arithmetic):
///
/// | frame | O | H | H | F |
/// |---|---|---|---|---|
/// | step 10 | (1.25, 2, 3) | (2.25, 2.5, 3.25) | (6.25, 7, 8) | (7, 7.25, 8.5) |
/// | step 20 | (9.875, 2, 3) | (10.75, 2.5, 3.25) | (6.5, −3, 8) | (7.25, −2.75, 8.5) |
/// | step 30 | (0.25, 2, 3) | (−0.75, 2.5, 3.25) | (6.25, 7, 28) | (7, 7.25, 28.5) |
///
/// Molecules (first frame, bond 1.6 Å, C-2 minimum image with L = 10): |O−H| = |(1, 0.5, 0.25)| =
/// 1.146 < 1.6 and |H−F| = |(0.75, 0.25, 0.5)| = 0.935 < 1.6; the closest cross pair (H at
/// (2.25, 2.5, 3.25), H at (6.25, 7, 8)) is |(4, 4.5, 4.75)| ≫ 1.6. So {0, 1} and {2, 3}.
///
/// Wrap shift per molecule = −floor(c/L)·L per component, c = centroid of the molecule's first
/// atom (0 for A, 2 for B):
/// - step 10: A (1.25, 2, 3) → 0; B (6.25, 7, 8) → 0.
/// - step 20: A (9.875, 2, 3) → 0 (the H centroid at x = 10.75 stays outside: the molecule
///   straddles x = L and is not split; O's bead 1 at x = 10.25 lies outside while its centroid is
///   inside: the ring polymer is not split); B (6.5, −3, 8) → (0, +10, 0).
/// - step 30: A (0.25, 2, 3) → 0 (H centroid at x = −0.75 stays); B (6.25, 7, 28) → (0, 0, −20)
///   (drifted by +2L in z).
pub const INPUT: &str = concat!(
    "8\n",
    "step=10 beads=2 atoms=4 units=angstrom\n",
    "O 1 2 3\n",
    "H 2 2.5 3\n",
    "H 6 7 8\n",
    "F 7 7 8.5\n",
    "O 1.5 2 3\n",
    "H 2.5 2.5 3.5\n",
    "H 6.5 7 8\n",
    "F 7 7.5 8.5\n",
    "8\n",
    "step=20 beads=2 atoms=4 units=angstrom\n",
    "O 9.5 2 3\n",
    "H 10.5 2.5 3\n",
    "H 6.25 -3.25 8\n",
    "F 7.25 -3 8.5\n",
    "O 10.25 2 3\n",
    "H 11 2.5 3.5\n",
    "H 6.75 -2.75 8\n",
    "F 7.25 -2.5 8.5\n",
    "8\n",
    "step=30 beads=2 atoms=4 units=angstrom\n",
    "O 0 2 3\n",
    "H -1 2.5 3\n",
    "H 6 7 27.75\n",
    "F 7 7 28.5\n",
    "O 0.5 2 3\n",
    "H -0.5 2.5 3.5\n",
    "H 6.5 7 28.25\n",
    "F 7 7.5 28.5\n",
);

/// The box of test 1, as passed to `--box-angstrom`.
pub const BOX: &str = "10";

/// Default output (centroids) with `--box-angstrom 10`: the wrapped centroids of the table above
/// (step 20: B + (0, 10, 0); step 30: B + (0, 0, −20)). Line 2: `step=<s>`, then the lattice
/// `Lattice="L 0 0 0 L 0 0 0 L"` with L written with `{}` ("10"), `pbc="T T T"`, then
/// `Properties=species:S:1:pos:R:3`, separated by single spaces.
pub const CENTROIDS: &str = concat!(
    "4\n",
    "step=10 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 1.25 2 3\n",
    "H 2.25 2.5 3.25\n",
    "H 6.25 7 8\n",
    "F 7 7.25 8.5\n",
    "4\n",
    "step=20 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 9.875 2 3\n",
    "H 10.75 2.5 3.25\n",
    "H 6.5 7 8\n",
    "F 7.25 7.25 8.5\n",
    "4\n",
    "step=30 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 0.25 2 3\n",
    "H -0.75 2.5 3.25\n",
    "H 6.25 7 8\n",
    "F 7 7.25 8.5\n",
);

/// `--beads --box-angstrom 10`: all P·N = 8 beads, bead-major, each shifted by its molecule's
/// vector (step 20: atoms 2, 3 + (0, 10, 0); step 30: atoms 2, 3 + (0, 0, −20)), then the bead
/// index 0 … P−1.
pub const BEADS: &str = concat!(
    "8\n",
    "step=10 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:bead:I:1\n",
    "O 1 2 3 0\n",
    "H 2 2.5 3 0\n",
    "H 6 7 8 0\n",
    "F 7 7 8.5 0\n",
    "O 1.5 2 3 1\n",
    "H 2.5 2.5 3.5 1\n",
    "H 6.5 7 8 1\n",
    "F 7 7.5 8.5 1\n",
    "8\n",
    "step=20 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:bead:I:1\n",
    "O 9.5 2 3 0\n",
    "H 10.5 2.5 3 0\n",
    "H 6.25 6.75 8 0\n",
    "F 7.25 7 8.5 0\n",
    "O 10.25 2 3 1\n",
    "H 11 2.5 3.5 1\n",
    "H 6.75 7.25 8 1\n",
    "F 7.25 7.5 8.5 1\n",
    "8\n",
    "step=30 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:bead:I:1\n",
    "O 0 2 3 0\n",
    "H -1 2.5 3 0\n",
    "H 6 7 7.75 0\n",
    "F 7 7 8.5 0\n",
    "O 0.5 2 3 1\n",
    "H -0.5 2.5 3.5 1\n",
    "H 6.5 7 8.25 1\n",
    "F 7 7.5 8.5 1\n",
);

/// `--every 2 --box-angstrom 10`: kept frames are those with index 0, 2, … in the input (counted
/// from 0), i.e. steps 10 and 30 of `CENTROIDS`.
pub const EVERY2: &str = concat!(
    "4\n",
    "step=10 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 1.25 2 3\n",
    "H 2.25 2.5 3.25\n",
    "H 6.25 7 8\n",
    "F 7 7.25 8.5\n",
    "4\n",
    "step=30 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 0.25 2 3\n",
    "H -0.75 2.5 3.25\n",
    "H 6.25 7 8\n",
    "F 7 7.25 8.5\n",
);

/// `--no-wrap --box-angstrom 10`: the raw centroids of the table above, with the lattice.
pub const NO_WRAP: &str = concat!(
    "4\n",
    "step=10 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 1.25 2 3\n",
    "H 2.25 2.5 3.25\n",
    "H 6.25 7 8\n",
    "F 7 7.25 8.5\n",
    "4\n",
    "step=20 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 9.875 2 3\n",
    "H 10.75 2.5 3.25\n",
    "H 6.5 -3 8\n",
    "F 7.25 -2.75 8.5\n",
    "4\n",
    "step=30 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
    "O 0.25 2 3\n",
    "H -0.75 2.5 3.25\n",
    "H 6.25 7 28\n",
    "F 7 7.25 28.5\n",
);

/// No box: the raw centroids, no `Lattice`/`pbc`.
pub const NO_BOX: &str = concat!(
    "4\n",
    "step=10 Properties=species:S:1:pos:R:3\n",
    "O 1.25 2 3\n",
    "H 2.25 2.5 3.25\n",
    "H 6.25 7 8\n",
    "F 7 7.25 8.5\n",
    "4\n",
    "step=20 Properties=species:S:1:pos:R:3\n",
    "O 9.875 2 3\n",
    "H 10.75 2.5 3.25\n",
    "H 6.5 -3 8\n",
    "F 7.25 -2.75 8.5\n",
    "4\n",
    "step=30 Properties=species:S:1:pos:R:3\n",
    "O 0.25 2 3\n",
    "H -0.75 2.5 3.25\n",
    "H 6.25 7 28\n",
    "F 7 7.25 28.5\n",
);

/// The five test 1 variants: (name, extra arguments, expected output).
pub const VARIANTS: [(&str, &[&str], &str); 5] = [
    ("centroids", &["--box-angstrom", BOX], CENTROIDS),
    ("beads", &["--beads", "--box-angstrom", BOX], BEADS),
    ("every2", &["--every", "2", "--box-angstrom", BOX], EVERY2),
    ("no_wrap", &["--no-wrap", "--box-angstrom", BOX], NO_WRAP),
    ("no_box", &[], NO_BOX),
];

pub fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().expect("temporary directory")
}

/// Writes `text` to `dir/name` and returns the path.
pub fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, text).expect("write input");
    p
}

/// Runs `pimd-traj <inputs>... --out <out> <args>`.
pub fn run_traj(inputs: &[&Path], out: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pimd-traj"))
        .args(inputs)
        .arg("--out")
        .arg(out)
        .args(args)
        .output()
        .expect("cannot start pimd-traj")
}

/// Runs the binary and returns the output file's text; panics with stderr on failure.
pub fn convert(inputs: &[&Path], out: &Path, args: &[&str]) -> String {
    let o = run_traj(inputs, out, args);
    assert!(
        o.status.success(),
        "pimd-traj {args:?} failed ({:?}): {}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    std::fs::read_to_string(out).expect("read output")
}

/// One extended-XYZ frame as read back by the tests: line 2 verbatim and the rows
/// (symbol, [x, y, z] Å, bead index if the `bead` column is present).
#[derive(Debug, Clone)]
pub struct XFrame {
    pub comment: String,
    pub rows: Vec<(String, [f64; 3], Option<usize>)>,
}

impl XFrame {
    /// The `step=<s>` value of line 2.
    pub fn step(&self) -> u64 {
        self.comment
            .split(' ')
            .find_map(|t| t.strip_prefix("step="))
            .unwrap_or_else(|| panic!("no step= in {:?}", self.comment))
            .parse()
            .unwrap()
    }
}

/// Reads the output of `pimd-traj` (a minimal reader for the layout of MILESTONE_9 M9b: count line,
/// comment line, rows `symbol x y z [bead]` with single spaces; every number written with `{}`).
pub fn read_extxyz(text: &str) -> Vec<XFrame> {
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut frames = Vec::new();
    while i < lines.len() {
        let n: usize = lines[i]
            .parse()
            .unwrap_or_else(|_| panic!("count line {:?}", lines[i]));
        let comment = lines[i + 1].to_string();
        let beads = comment.ends_with(":bead:I:1");
        let rows = lines[i + 2..i + 2 + n]
            .iter()
            .map(|l| {
                let w: Vec<&str> = l.split(' ').collect();
                assert_eq!(w.len(), if beads { 5 } else { 4 }, "row {l:?}");
                let x = [1, 2, 3].map(|k| {
                    let v: f64 = w[k].parse().unwrap();
                    assert_eq!(format!("{v}"), w[k], "not written with {{}}: {l:?}");
                    v
                });
                (w[0].to_string(), x, beads.then(|| w[4].parse().unwrap()))
            })
            .collect();
        frames.push(XFrame { comment, rows });
        i += 2 + n;
    }
    frames
}

/// One input frame: (step, P·N rows bead-major of (symbol, [x, y, z] Å)).
pub type InFrame<'a> = (u64, Vec<(&'a str, [f64; 3])>);

/// Writes frames in the `trajectory.xyz` layout (MILESTONE_5): `frames` = (step, P·N rows
/// bead-major of (symbol, [x, y, z])), numbers with `{}`.
pub fn trajectory_text(p: usize, n: usize, frames: &[InFrame]) -> String {
    let mut s = String::new();
    for (step, rows) in frames {
        assert_eq!(rows.len(), p * n, "test setup");
        s.push_str(&format!(
            "{}\nstep={step} beads={p} atoms={n} units=angstrom\n",
            p * n
        ));
        for (sym, x) in rows {
            s.push_str(&format!("{sym} {} {} {}\n", x[0], x[1], x[2]));
        }
    }
    s
}

/// Q40: a failed conversion exits with a non-zero status that is not a panic (Rust's panic exit
/// code is 101), and reports on stderr.
pub fn expect_clean_failure(o: &Output, what: &str) -> String {
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!o.status.success(), "{what}: expected failure, got success");
    assert!(
        o.status.code().is_some_and(|c| c != 101) && !err.contains("panicked"),
        "{what}: must fail with an error, not a panic ({:?}): {err}",
        o.status
    );
    assert!(!err.trim().is_empty(), "{what}: empty stderr");
    err
}

/// True if `msg` names line `k` of `path`: as `line <k>` (any case), or `<path>:<k>`, with no digit
/// following (Q40).
pub fn mentions_line(msg: &str, path: &Path, k: usize) -> bool {
    let p = path.display().to_string();
    let lower = msg.to_lowercase();
    let pats = [format!("line {k}"), format!("{}:{k}", p.to_lowercase())];
    pats.iter().any(|pat| {
        lower.match_indices(pat.as_str()).any(|(i, _)| {
            !lower[i + pat.len()..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        })
    })
}

/// Q40: the message names the file (its path as given) and one of the accepted 1-based line
/// numbers `lines`.
pub fn assert_names(msg: &str, path: &Path, lines: &[usize], what: &str) {
    assert!(
        msg.contains(&path.display().to_string()),
        "{what}: message must name the file {}: {msg}",
        path.display()
    );
    assert!(
        lines.iter().any(|&k| mentions_line(msg, path, k)),
        "{what}: message must name line {lines:?} of {}: {msg}",
        path.display()
    );
}
