//! Config additions of M8a (MILESTONE_8 "M8a: Config"; SPEC U-3, D-3 chunk-size sentence): the
//! xyz positions file `system.positions_xyz`, and `potential.beads_per_graph` with kinds other than
//! "mace". Runs without the feature `torch` (Lennard-Jones and socket configs); the "mace" side is
//! in `config_m8_mace.rs`.
//!
//! `positions_xyz` (brief): standard xyz, line 1 the atom count, line 2 a comment (ignored), then
//! one line `symbol x y z` (Å) per atom; extra columns are an error. Read at parse time, the path
//! as given. It replaces `positions_angstrom` and `symbols` (either alongside it is `Parse` naming
//! that key). An unreadable file is `CliError::Io` naming `system.positions_xyz` and the path;
//! malformed contents, a count line ≠ the number of atom lines, a count ≠ `n_atoms` or a non-finite
//! coordinate are `InvalidValue("system.positions_xyz")`. The parsed positions and symbols hash as
//! if given inline (not the path), and runs of the two are byte-identical.
//!
//! Assumptions (reported): columns are separated by any run of spaces (as written by common xyz
//! writers, e.g. aligned columns); every file here ends with one newline after the last atom line;
//! lines after the last atom line are not tested. Å → bohr is the U-2 literal division
//! x / 0.529177210903, which is what `positions_angstrom` does, so the parsed values are bitwise
//! those of the inline config.

mod common;

use std::path::{Path, PathBuf};

use common::{BASE, bytes, parse, set, text, tmp};
use pimd_cli::{CliError, Config, run};

const BOHR_A: f64 = 0.529177210903; // Å per bohr (U-2)
const N: usize = 32;

fn err(toml: &str) -> CliError {
    match Config::from_toml_str(toml) {
        Ok(c) => panic!("config accepted: {c:?}\n{toml}"),
        Err(e) => e,
    }
}

/// Rejected with exactly `InvalidValue { what: "system.positions_xyz" }`.
fn invalid_xyz(toml: &str, what: &str) {
    match err(toml) {
        CliError::InvalidValue { what: field, .. } => {
            assert_eq!(field, "system.positions_xyz", "{what}")
        }
        other => panic!("{what}: expected InvalidValue(system.positions_xyz), got {other:?}"),
    }
}

/// `Parse` whose message mentions `key`.
fn parse_naming(toml: &str, key: &str) {
    match err(toml) {
        CliError::Parse(m) => assert!(m.contains(key), "Parse message {m:?} must name {key}"),
        other => panic!("expected Parse naming {key}, got {other:?}"),
    }
}

/// The 32 positions of the test files, Å, as decimal strings (10 decimals, not exactly
/// representable): a 4 × 4 × 2 grid of spacing 3.125 Å with small, distinct offsets, so atom order
/// matters and every coordinate differs.
fn coords() -> Vec<[String; 3]> {
    (0..N)
        .map(|i| {
            let (ix, iy, iz) = (i % 4, (i / 4) % 4, i / 16);
            let f = |g: usize, k: usize| {
                format!(
                    "{:.10}",
                    (g as f64 + 0.5) * 3.125 + 0.0123456789 * ((i * 3 + k) % 7) as f64 - 0.03
                )
            };
            [f(ix, 0), f(iy, 1), f(iz, 2)]
        })
        .collect()
}

/// Element labels of the 32 atoms (output only for LJ): "Ne" except a few isotope-style labels,
/// so that symbols are not all equal and their order is checked.
fn labels() -> Vec<String> {
    (0..N)
        .map(|i| match i % 5 {
            0 => "Ne".to_string(),
            1 => "Ne20".to_string(),
            2 => "Ne".to_string(),
            3 => "X".to_string(),
            _ => "Ne22".to_string(),
        })
        .collect()
}

/// A well-formed xyz text: count line, an extended-xyz style comment, then the atom lines, with
/// runs of spaces between columns on some lines.
fn good_xyz() -> String {
    let mut s =
        format!("{N}\nLattice=\"12.5 0 0 0 12.5 0 0 0 12.5\" Properties=species:S:1:pos:R:3 x\n");
    for (i, (l, c)) in labels().iter().zip(coords()).enumerate() {
        if i % 3 == 0 {
            s.push_str(&format!("{l}    {}   {}  {}\n", c[0], c[1], c[2]));
        } else {
            s.push_str(&format!("{l} {} {} {}\n", c[0], c[1], c[2]));
        }
    }
    s
}

/// Writes `contents` to `dir/name` and returns the path.
fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, contents).unwrap();
    p
}

/// BASE (32 Ne atoms, LJ, box 12.5 Å) with `positions_xyz = path`.
fn with_xyz(path: &Path) -> String {
    set(
        BASE,
        "system",
        "positions_xyz",
        &format!("\"{}\"", path.display()),
    )
}

/// BASE with the same positions and symbols given inline.
fn inline() -> String {
    let pos: Vec<String> = coords()
        .iter()
        .map(|c| format!("[{}, {}, {}]", c[0], c[1], c[2]))
        .collect();
    let sym: Vec<String> = labels().iter().map(|l| format!("\"{l}\"")).collect();
    let t = set(
        BASE,
        "system",
        "positions_angstrom",
        &format!("[{}]", pos.join(", ")),
    );
    set(&t, "system", "symbols", &format!("[{}]", sym.join(", ")))
}

/// `good_xyz()` with atom line `k` (0-based) replaced by `line`.
fn with_atom_line(k: usize, line: &str) -> String {
    let mut lines: Vec<String> = good_xyz().lines().map(str::to_string).collect();
    lines[2 + k] = line.to_string();
    lines.join("\n") + "\n"
}

/// U-3, MILESTONE_8 "Config": a well-formed file gives positions x_Å / 0.529177210903 (bitwise,
/// in file order) and the file's symbols (in order); the whole `Config` equals the inline
/// equivalent (`positions_angstrom` + `symbols`).
#[test]
fn xyz_positions_and_symbols() {
    let d = tmp();
    let c = parse(&with_xyz(&write(d.path(), "box.xyz", &good_xyz())));
    let pos = c
        .system
        .positions
        .as_ref()
        .expect("positions from the file");
    assert_eq!(pos.len(), N);
    for (i, (p, s)) in pos.iter().zip(coords()).enumerate() {
        for k in 0..3 {
            let want = s[k].parse::<f64>().unwrap() / BOHR_A;
            assert_eq!(
                p[k].to_bits(),
                want.to_bits(),
                "atom {i} component {k}: {} bohr vs hand {want}",
                p[k]
            );
        }
    }
    assert_eq!(c.system.symbols, Some(labels()));
    assert_eq!(
        c,
        parse(&inline()),
        "xyz config differs from its inline equivalent"
    );
}

/// MILESTONE_8 "Config": the comment line is ignored (two files differing only there parse
/// equal); the file is found by the path as given; a gas-phase config (no box) takes its positions
/// from the file.
#[test]
fn xyz_comment_ignored_and_gas_phase() {
    let d = tmp();
    let a = write(d.path(), "a.xyz", &good_xyz());
    let other = good_xyz().replacen(
        &good_xyz().lines().nth(1).unwrap().to_string(),
        "anything at all: 1 2 3 4 5",
        1,
    );
    let b = write(d.path(), "b.xyz", &other);
    assert_eq!(parse(&with_xyz(&a)), parse(&with_xyz(&b)));
    let mut gas = common::remove(&with_xyz(&a), "box_angstrom");
    gas = common::remove(&gas, "rdf_bins");
    gas = common::remove(&gas, "rdf_rmax_angstrom");
    let c = parse(&gas);
    assert!(c.system.cell.is_none());
    assert_eq!(c.system.positions, parse(&inline()).system.positions);
}

/// MILESTONE_8 "Config": a count line that differs from the number of atom lines (31 or 33 for 32
/// lines; 32 for 31 or 33 lines), that is not a non-negative integer, or a count equal to the atom
/// lines but ≠ `n_atoms` → `InvalidValue("system.positions_xyz")`.
#[test]
fn xyz_wrong_count() {
    let d = tmp();
    let body: Vec<String> = good_xyz().lines().skip(1).map(str::to_string).collect();
    let file = |count: &str, atoms: usize| {
        let mut s = format!("{count}\n");
        for l in body.iter().take(1 + atoms) {
            s.push_str(l);
            s.push('\n');
        }
        s
    };
    let extra = format!("{}\nNe 0.1 0.2 0.3\n", good_xyz().trim_end());
    let cases = [
        ("count 31, 32 lines", file("31", 32)),
        ("count 33, 32 lines", file("33", 32)),
        ("count 32, 31 lines", file("32", 31)),
        ("count 32, 33 lines", extra),
        ("count 0", file("0", 32)),
        ("count -32", file("-32", 32)),
        ("count 32.0", file("32.0", 32)),
        ("count abc", file("abc", 32)),
        ("empty count line", file("", 32)),
        ("count and more on line 1", file("32 atoms", 32)),
        ("empty file", String::new()),
        ("count only", "32\n".to_string()),
    ];
    for (what, contents) in cases {
        invalid_xyz(&with_xyz(&write(d.path(), "bad.xyz", &contents)), what);
    }
    // 31 atoms, count 31: consistent file, but n_atoms = 32.
    invalid_xyz(
        &with_xyz(&write(d.path(), "short.xyz", &file("31", 31))),
        "count 31 = lines, n_atoms 32",
    );
    // The same file with n_atoms = 31 is accepted (control): masses come from mass_amu.
    let t = set(
        &with_xyz(&write(d.path(), "short.xyz", &file("31", 31))),
        "system",
        "n_atoms",
        "31",
    );
    assert_eq!(parse(&t).system.positions.unwrap().len(), 31);
}

/// MILESTONE_8 "Config": an atom line with an extra column, a missing column, a non-numeric
/// coordinate or no symbol → `InvalidValue("system.positions_xyz")` (first, middle and last atom).
#[test]
fn xyz_malformed_atom_lines() {
    let d = tmp();
    for k in [0, 17, N - 1] {
        for (what, line) in [
            ("extra column", "Ne 1.0 2.0 3.0 4.0"),
            ("extra text column", "Ne 1.0 2.0 3.0 x"),
            ("missing column", "Ne 1.0 2.0"),
            ("symbol only", "Ne"),
            ("empty line", ""),
            ("non-numeric", "Ne 1.0 abc 3.0"),
            ("no symbol", "1.0 2.0 3.0"),
            ("comma separated", "Ne,1.0,2.0,3.0"),
        ] {
            invalid_xyz(
                &with_xyz(&write(d.path(), "bad.xyz", &with_atom_line(k, line))),
                &format!("atom {k}: {what}"),
            );
        }
    }
}

/// MILESTONE_8 "Config": a non-finite coordinate (nan, NaN, inf, -inf, infinity, 1e999 which
/// overflows to inf) → `InvalidValue("system.positions_xyz")`.
#[test]
fn xyz_non_finite() {
    let d = tmp();
    for v in ["nan", "NaN", "inf", "-inf", "infinity", "1e999"] {
        for k in 0..3 {
            let mut c = ["1.0", "2.0", "3.0"];
            c[k] = v;
            let line = format!("Ne {} {} {}", c[0], c[1], c[2]);
            invalid_xyz(
                &with_xyz(&write(d.path(), "bad.xyz", &with_atom_line(5, &line))),
                &format!("{v} in column {k}"),
            );
        }
    }
}

/// MILESTONE_8 "Config" (Q24 precedent): a file that cannot be read is `CliError::Io` whose
/// message names `system.positions_xyz` and the path (missing file, and a directory).
#[test]
fn xyz_unreadable_is_io_error() {
    let d = tmp();
    for path in [d.path().join("no_such_file.xyz"), d.path().to_path_buf()] {
        match Config::from_toml_str(&with_xyz(&path)) {
            Err(CliError::Io(e)) => {
                let m = e.to_string();
                assert!(
                    m.contains("system.positions_xyz"),
                    "message {m:?} must name system.positions_xyz"
                );
                assert!(
                    m.contains(&path.display().to_string()),
                    "message {m:?} must name the path"
                );
            }
            other => panic!("{}: expected CliError::Io, got {other:?}", path.display()),
        }
    }
}

/// MILESTONE_8 "Config": `positions_xyz` replaces `positions_angstrom` and `symbols`; either of
/// them alongside it is `Parse` naming that key.
#[test]
fn xyz_excludes_inline_positions_and_symbols() {
    let d = tmp();
    let x = with_xyz(&write(d.path(), "box.xyz", &good_xyz()));
    let both = inline();
    let pos_line = both
        .lines()
        .find(|l| l.starts_with("positions_angstrom"))
        .unwrap();
    let sym_line = both.lines().find(|l| l.starts_with("symbols")).unwrap();
    let t = x.replacen("[system]\n", &format!("[system]\n{pos_line}\n"), 1);
    parse_naming(&t, "positions_angstrom");
    let t = x.replacen("[system]\n", &format!("[system]\n{sym_line}\n"), 1);
    parse_naming(&t, "symbols");
}

/// The `config hash = …` line of a run.log.
fn hash_line(dir: &Path) -> String {
    text(&dir.join("run.log"))
        .lines()
        .find(|l| l.starts_with("config hash = "))
        .expect("config hash line")
        .to_string()
}

/// Runs `toml` into a fresh directory.
fn run_dir(toml: &str) -> tempfile::TempDir {
    let d = tmp();
    run(&parse(toml), d.path()).unwrap_or_else(|e| panic!("run failed: {e:?}"));
    d
}

/// MILESTONE_8 "Config" (hash): a config with `positions_xyz` hashes as its inline equivalent,
/// not by the path (the same file at two paths gives the same hash), and its run is
/// byte-identical to the inline run (observables.csv, rdf.csv, checkpoint.bin, run.log). Changing
/// one coordinate in the file changes the hash (the positions enter it).
#[test]
fn xyz_hash_and_run_equal_inline() {
    let d = tmp();
    let a = write(d.path(), "a.xyz", &good_xyz());
    std::fs::create_dir(d.path().join("sub")).unwrap();
    let b = write(&d.path().join("sub"), "renamed.xyz", &good_xyz());
    let ra = run_dir(&with_xyz(&a));
    let rb = run_dir(&with_xyz(&b));
    let ri = run_dir(&inline());
    assert_eq!(hash_line(ra.path()), hash_line(ri.path()), "xyz vs inline");
    assert_eq!(
        hash_line(ra.path()),
        hash_line(rb.path()),
        "path must not be hashed"
    );
    for f in ["observables.csv", "rdf.csv", "checkpoint.bin", "run.log"] {
        assert_eq!(
            bytes(&ra.path().join(f)),
            bytes(&ri.path().join(f)),
            "{f}: xyz run differs from the inline run"
        );
    }
    let c = coords();
    let moved = with_atom_line(3, &format!("X {} {} {}", c[3][0], c[3][1], "0.1234567890"));
    let rm = run_dir(&with_xyz(&write(d.path(), "moved.xyz", &moved)));
    assert_ne!(hash_line(ra.path()), hash_line(rm.path()));
}

/// MILESTONE_8 "Config": `beads_per_graph` belongs only to kind = "mace": with kind = "lj" or
/// "socket" it is `Parse` naming the key.
#[test]
fn beads_per_graph_rejected_with_other_kinds() {
    parse_naming(
        &set(BASE, "potential", "beads_per_graph", "2"),
        "beads_per_graph",
    );
    let sock = common::water_toml("unix:/tmp/pimd-m8-unused.sock", 1);
    parse(&sock);
    parse_naming(
        &set(&sock, "potential", "beads_per_graph", "2"),
        "beads_per_graph",
    );
}
