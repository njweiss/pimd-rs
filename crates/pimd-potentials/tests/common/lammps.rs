//! Test-only reader for the LAMMPS reference data `data/lj_snapshot.json` (written by
//! `data/gen_lj_snapshot.py`; LAMMPS version recorded in the file), shared by the M3 tests of
//! `pimd-potentials`, `pimd-integrators` and `pimd-estimators` (the latter two include this file
//! with `#[path]`).
//!
//! Units: LAMMPS `units lj` with ε = σ = m = 1 is the engine's atomic units with ε = 1 Eh,
//! σ = 1 bohr, m = 1 m_e (time unit σ√(m/ε) = ħ/Eh, pressure ε/σ³ = Eh/bohr³), so every number is
//! used as is, without conversion (see the generator's docstring).
//!
//! The JSON reader is a minimal recursive-descent parser for generator-written files (no JSON
//! dependency). Numbers are parsed with `str::parse::<f64>`, which is correctly rounded, so the
//! shortest round-trip representations written by Python give back the exact doubles.
#![allow(dead_code)]

use pimd_core::{Beads, Cell};

/// A parsed JSON value.
#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parses a complete JSON document; panics on malformed input.
    pub fn parse(text: &str) -> Json {
        let b = text.as_bytes();
        let mut pos = 0;
        let v = value(b, &mut pos);
        skip_ws(b, &mut pos);
        assert_eq!(pos, b.len(), "trailing characters in JSON");
        v
    }

    /// Member `key` of an object; panics if absent.
    pub fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(m) => m
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("JSON key {key} missing")),
            _ => panic!("JSON value is not an object (looking up {key})"),
        }
    }

    /// Follows a path of object keys.
    pub fn at(&self, path: &[&str]) -> &Json {
        path.iter().fold(self, |v, k| v.get(k))
    }

    pub fn num(&self) -> f64 {
        match self {
            Json::Num(x) => *x,
            other => panic!("JSON value is not a number: {other:?}"),
        }
    }

    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            other => panic!("JSON value is not an array: {other:?}"),
        }
    }

    /// An array of numbers.
    pub fn nums(&self) -> Vec<f64> {
        self.arr().iter().map(Json::num).collect()
    }
}

fn skip_ws(b: &[u8], pos: &mut usize) {
    while *pos < b.len() && b[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

fn expect(b: &[u8], pos: &mut usize, lit: &str) {
    assert!(
        b[*pos..].starts_with(lit.as_bytes()),
        "JSON: expected {lit} at byte {pos}"
    );
    *pos += lit.len();
}

fn string(b: &[u8], pos: &mut usize) -> String {
    expect(b, pos, "\"");
    let start = *pos;
    while b[*pos] != b'"' {
        assert_ne!(b[*pos], b'\\', "JSON: escapes are not supported");
        *pos += 1;
    }
    let s = std::str::from_utf8(&b[start..*pos]).unwrap().to_string();
    *pos += 1;
    s
}

fn value(b: &[u8], pos: &mut usize) -> Json {
    skip_ws(b, pos);
    match b[*pos] {
        b'{' => {
            *pos += 1;
            let mut members = Vec::new();
            skip_ws(b, pos);
            if b[*pos] == b'}' {
                *pos += 1;
                return Json::Obj(members);
            }
            loop {
                skip_ws(b, pos);
                let k = string(b, pos);
                skip_ws(b, pos);
                expect(b, pos, ":");
                members.push((k, value(b, pos)));
                skip_ws(b, pos);
                if b[*pos] == b',' {
                    *pos += 1;
                } else {
                    expect(b, pos, "}");
                    return Json::Obj(members);
                }
            }
        }
        b'[' => {
            *pos += 1;
            let mut items = Vec::new();
            skip_ws(b, pos);
            if b[*pos] == b']' {
                *pos += 1;
                return Json::Arr(items);
            }
            loop {
                items.push(value(b, pos));
                skip_ws(b, pos);
                if b[*pos] == b',' {
                    *pos += 1;
                } else {
                    expect(b, pos, "]");
                    return Json::Arr(items);
                }
            }
        }
        b'"' => Json::Str(string(b, pos)),
        b't' => {
            expect(b, pos, "true");
            Json::Bool(true)
        }
        b'f' => {
            expect(b, pos, "false");
            Json::Bool(false)
        }
        b'n' => {
            expect(b, pos, "null");
            Json::Null
        }
        _ => {
            let start = *pos;
            while *pos < b.len()
                && matches!(b[*pos], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                *pos += 1;
            }
            let s = std::str::from_utf8(&b[start..*pos]).unwrap();
            Json::Num(s.parse().unwrap_or_else(|_| panic!("JSON: bad number {s}")))
        }
    }
}

/// The text of `lj_snapshot.json`.
pub const SNAPSHOT_JSON: &str = include_str!("../data/lj_snapshot.json");

/// The parsed snapshot file.
pub fn snapshot_json() -> Json {
    Json::parse(SNAPSHOT_JSON)
}

/// The 256-atom LJ snapshot (ρ* = 0.8, T* = 1) and its LJ parameters, in engine units.
pub struct Snapshot {
    pub n: usize,
    pub box_l: f64,
    pub epsilon: f64,
    pub sigma: f64,
    pub mass: f64,
    pub cutoff: f64,
    pub skin: f64,
    /// `[1, N, 3]`, bohr, wrapped into [0, L).
    pub q: Beads,
    /// Momenta m v, `[1, N, 3]`.
    pub p: Beads,
}

impl Snapshot {
    pub fn load() -> Snapshot {
        let j = snapshot_json();
        let n = j.get("n_atoms").num() as usize;
        let box_l = j.get("box").num();
        let par = j.get("parameters");
        let mass = par.get("mass").num();
        let q = Beads::from_vec(1, n, j.get("positions").nums()).unwrap();
        let v = j.get("velocities").nums();
        let p = Beads::from_vec(1, n, v.iter().map(|v| mass * v).collect()).unwrap();
        Snapshot {
            n,
            box_l,
            epsilon: par.get("epsilon").num(),
            sigma: par.get("sigma").num(),
            mass,
            cutoff: par.get("cutoff").num(),
            skin: par.get("skin").num(),
            q,
            p,
        }
    }

    pub fn cell(&self) -> Cell {
        Cell::new([self.box_l; 3]).unwrap()
    }

    pub fn masses(&self) -> Vec<f64> {
        vec![self.mass; self.n]
    }
}

/// Largest |a − b| / max(|b|, floor) over two equal-length slices.
pub fn max_rel(a: &[f64], b: &[f64], floor: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(floor))
        .fold(0.0, f64::max)
}
