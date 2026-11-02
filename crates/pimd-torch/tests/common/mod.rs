//! Test-only helpers for the `pimd-torch` tests (MILESTONE_6 "M6b: Required tests").
//!
//! Reference data: `data/mace_ref.json`, written by `data/gen_mace_ref.py` (Python MACE
//! `MACECalculator` on the committed tiny model `data/tiny_mace.model`, mace's own neighbour
//! list, float64, CPU). The adapter under test loads the committed float64 export
//! `data/tiny_mace_f64.pt` of that model (`data/gen_tiny_mace.py`).
//!
//! Unit conversions here use the SPEC U-2 literals (CODATA 2018), written out, not read from
//! `pimd_core::units`: E [Eh] = E [eV] / 27.211386245988; F [Eh/bohr] = F [eV/Å] · 0.529177210903
//! / 27.211386245988. Positions are stored in bohr in the JSON (the generator evaluated MACE at
//! q_bohr · 0.529177210903 Å, which is the adapter's conversion, P-6).
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

/// The minimal JSON reader of the M3 tests (no JSON dependency).
#[path = "../../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

use std::path::PathBuf;

pub use lammps::Json;
use pimd_core::{Beads, Cell};
use pimd_potentials::Evaluation;
use pimd_torch::{Device, Dtype, MacePotential};

/// One bohr in Å (U-2).
pub const BOHR_A: f64 = 0.529177210903;
/// One hartree in eV (U-2, Q13).
pub const EV_PER_EH: f64 = 27.211386245988;

/// The test data directory.
pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// The committed float64 TorchScript export of the tiny model.
pub fn export_path() -> PathBuf {
    data_dir().join("tiny_mace_f64.pt")
}

/// The committed pickled tiny model (a MACE model file, not TorchScript).
pub fn pickled_model_path() -> PathBuf {
    data_dir().join("tiny_mace.model")
}

/// The parsed reference JSON.
pub fn reference() -> Json {
    let text = std::fs::read_to_string(data_dir().join("mace_ref.json")).expect("mace_ref.json");
    Json::parse(&text)
}

/// A reference system: geometry in bohr and the Python MACE results converted to atomic units.
pub struct System {
    pub name: String,
    pub symbols: Vec<String>,
    pub z: Vec<u8>,
    pub p: usize,
    pub n: usize,
    pub cell: Option<Cell>,
    /// Base (undisplaced) geometry, bohr, length N.
    pub base: Vec<[f64; 3]>,
    /// Bead positions `[P, N, 3]`, bohr (unwrapped for the box).
    pub q: Beads,
    /// Python MACE energies per bead, Eh.
    pub energies: Vec<f64>,
    /// Python MACE forces `[P, N, 3]`, Eh/bohr.
    pub forces: Beads,
    /// Python MACE energies per bead, eV, as stored.
    pub energies_ev: Vec<f64>,
}

fn strings(j: &Json) -> Vec<String> {
    j.arr()
        .iter()
        .map(|s| match s {
            Json::Str(s) => s.clone(),
            other => panic!("expected a string, got {other:?}"),
        })
        .collect()
}

/// `name` = "water", "ethanol", "waterbox" (P = 8 each) or "water_single" (P = 1).
pub fn system(name: &str) -> System {
    let r = reference();
    let s = r.at(&["systems", name]);
    let p = s.get("n_beads").num() as usize;
    let n = s.get("n_atoms").num() as usize;
    let cell = match s.get("cell_bohr") {
        Json::Null => None,
        c => {
            let l = c.nums();
            Some(Cell::new([l[0], l[1], l[2]]).unwrap())
        }
    };
    let base_flat = s.get("base_positions_bohr").nums();
    let base = (0..n)
        .map(|i| [base_flat[3 * i], base_flat[3 * i + 1], base_flat[3 * i + 2]])
        .collect();
    let q = Beads::from_vec(p, n, s.get("positions_bohr").nums()).unwrap();
    let energies_ev = s.get("energies_ev").nums();
    let energies = energies_ev.iter().map(|e| e / EV_PER_EH).collect();
    let f: Vec<f64> = s
        .get("forces_ev_per_angstrom")
        .nums()
        .iter()
        .map(|f| f * BOHR_A / EV_PER_EH)
        .collect();
    System {
        name: name.to_string(),
        symbols: strings(s.get("symbols")),
        z: s.get("atomic_numbers")
            .nums()
            .iter()
            .map(|&z| z as u8)
            .collect(),
        p,
        n,
        cell,
        base,
        q,
        energies,
        forces: Beads::from_vec(p, n, f).unwrap(),
        energies_ev,
    }
}

/// The P = 8 reference systems of the brief: water, ethanol, the 4-water box.
pub const SYSTEMS: [&str; 3] = ["water", "ethanol", "waterbox"];

/// Loads the tiny export for `z`, deterministic switch on (the config default).
pub fn load(z: &[u8], device: Device, dtype: Dtype, threads: usize) -> MacePotential {
    MacePotential::load(&export_path(), z, device, dtype, threads, true)
        .unwrap_or_else(|e| panic!("load failed: {e:?}"))
}

/// Evaluates all beads, panicking on error.
pub fn eval(pot: &mut MacePotential, q: &Beads, cell: Option<&Cell>) -> Evaluation {
    use pimd_potentials::Potential;
    pot.evaluate(q, cell, None)
        .unwrap_or_else(|e| panic!("evaluate failed: {e:?}"))
}

/// Bead j of `x` (`[N, 3]` values, flat).
pub fn bead(x: &Beads, j: usize) -> &[f64] {
    let n3 = x.n_atoms() * 3;
    &x.as_slice()[j * n3..(j + 1) * n3]
}

pub fn max_abs(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0.0, |m, x| m.max(x.abs()))
}

/// Worst per-bead errors of `got` against (`e_ref`, `f_ref`): (max_j |ΔE_j| / |E_j|,
/// max_j |ΔE_j| (Eh), max_j max|ΔF_j| / max|F_j|), with F_j the reference forces of bead j.
pub fn errors(got: &Evaluation, e_ref: &[f64], f_ref: &Beads) -> (f64, f64, f64) {
    let p = e_ref.len();
    assert_eq!(got.energies.len(), p, "one energy per bead (S-4)");
    assert_eq!(got.forces.n_beads(), p, "forces shape [P, N, 3]");
    assert_eq!(
        got.forces.n_atoms(),
        f_ref.n_atoms(),
        "forces shape [P, N, 3]"
    );
    let (mut er, mut ea, mut fr) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..p {
        let de = (got.energies[j] - e_ref[j]).abs();
        er = er.max(de / e_ref[j].abs());
        ea = ea.max(de);
        let a = bead(&got.forces, j);
        let b = bead(f_ref, j);
        let df = max_abs(a.iter().zip(b).map(|(x, y)| x - y));
        fr = fr.max(df / max_abs(b.iter().copied()));
    }
    (er, ea, fr)
}

/// Bitwise equality of energies and forces (f64 bit patterns).
pub fn bitwise_eq(a: &Evaluation, b: &Evaluation) -> bool {
    a.energies.len() == b.energies.len()
        && a.energies
            .iter()
            .zip(&b.energies)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.forces.as_slice().len() == b.forces.as_slice().len()
        && a.forces
            .as_slice()
            .iter()
            .zip(b.forces.as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Largest difference between two evaluations: (max |ΔE| Eh, max |ΔF| Eh/bohr).
pub fn max_diff(a: &Evaluation, b: &Evaluation) -> (f64, f64) {
    (
        max_abs(a.energies.iter().zip(&b.energies).map(|(x, y)| x - y)),
        max_abs(
            a.forces
                .as_slice()
                .iter()
                .zip(b.forces.as_slice())
                .map(|(x, y)| x - y),
        ),
    )
}

/// SplitMix64 (Steele, Lea, Flood 2014): seeded test-only generator.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [-a, a).
    pub fn sym(&mut self, a: f64) -> f64 {
        a * (2.0 * self.next_f64() - 1.0)
    }
}

/// P beads of `base` (bohr), each atom displaced uniformly by up to ±`amp` bohr per component.
pub fn displaced(base: &[[f64; 3]], p: usize, amp: f64, seed: u64) -> Beads {
    let mut rng = SplitMix64(seed);
    let n = base.len();
    let mut q = Beads::zeros(p, n);
    for j in 0..p {
        for (i, b) in base.iter().enumerate() {
            q.set(j, i, b.map(|x| x + rng.sym(amp)));
        }
    }
    q
}

/// Bead j of `q` as a one-bead `Beads`.
pub fn single(q: &Beads, j: usize) -> Beads {
    Beads::from_vec(1, q.n_atoms(), bead(q, j).to_vec()).unwrap()
}

/// Whether libtorch sees a CUDA device (tch, the crate's own backend dependency).
pub fn cuda_available() -> bool {
    tch::Cuda::is_available()
}
