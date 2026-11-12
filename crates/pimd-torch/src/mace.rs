//! `MacePotential` (SPEC P-6): a TorchScript MACE export (`scripts/export_mace.py`) evaluated
//! in-process, the P beads in consecutive chunks of c beads, each chunk one batched graph.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use pimd_core::units::{BOHR_IN_ANGSTROM, HARTREE_IN_EV};
use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, Potential, PotentialError};
use rayon::prelude::*;
use tch::{CModule, IValue, Kind, TchError, Tensor};

use crate::ffi;

/// Where the model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// libtorch on the CPU, with the intra-op thread count given to `MacePotential::load`.
    Cpu,
    /// CUDA device with this index.
    Cuda(usize),
}

/// Precision inside the model (D-1: f32 only inside the adapter; inputs and outputs are f64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// float32 inside the model; positions are cast on the way in, results back to f64.
    F32,
    /// float64 throughout (the export's own dtype).
    F64,
}

/// Element symbols H (Z = 1) through I (Z = 53), in order.
const ELEMENTS: [&str; 53] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I",
];

/// Atomic number of an element symbol ("H" -> 1, ...), H through I; `None` if unknown.
pub fn atomic_number(symbol: &str) -> Option<u8> {
    ELEMENTS
        .iter()
        .position(|&s| s == symbol)
        .and_then(|k| u8::try_from(k + 1).ok())
}

/// A libtorch error as `PotentialError::Backend`, prefixed with what was being done.
fn backend(what: &str, e: TchError) -> PotentialError {
    PotentialError::Backend(format!("{what}: {e}"))
}

/// A MACE model loaded from a TorchScript export, for a fixed list of N atoms (P-6).
pub struct MacePotential {
    /// The scripted `PimdMace` wrapper, on `device`, cast to `kind`.
    module: CModule,
    /// Species index (position in the model's `atomic_numbers()`) of each of the N atoms.
    species: Vec<i64>,
    /// Model cutoff r_max, bohr.
    r_max: f64,
    /// libtorch device.
    device: tch::Device,
    /// libtorch dtype inside the model (`Kind::Double` or `Kind::Float`).
    kind: Kind,
    /// libtorch intra-op thread count, applied on the calling thread by every `evaluate`.
    threads: i32,
    /// Edge buffers, cleared and refilled by every `evaluate`.
    edges: Edges,
    /// Input tensors that depend only on the bead count of a graph, for at most two graph sizes
    /// (a full chunk and the last, shorter one); the oldest entry is evicted first, so after a
    /// change of P or c it may hold a size the last call did not use (costs one rebuild only).
    constants: Vec<Constants>,
    /// Beads per graph c set by `with_beads_per_graph`; `None` = all beads in one graph (P-6).
    beads_per_graph: Option<usize>,
    /// v2 export: Σ_i E0(Z_i) over the N atoms (Eh), added to every bead's interaction energy;
    /// `None` for a v1 export, whose energy is used as returned (P-6).
    e0_sum: Option<f64>,
}

impl MacePotential {
    /// Loads a TorchScript export (export_mace.py) for N = `atomic_numbers.len()` atoms. Only
    /// after the file, the module, its dtype and the atomic numbers are validated does it change
    /// libtorch's process-wide settings: the intra-op thread count is set to `threads`, the
    /// TorchScript profiling executor disabled and the deterministic-algorithms switch set to
    /// `deterministic`. A failed load leaves them unchanged. libtorch keeps the thread count per
    /// OS thread, so `evaluate` applies `threads` again on its calling thread (P-6).
    ///
    /// A v2 export (`interface_version()` = 2) returns the interaction energy; its reference
    /// energies `atomic_energies()` (eV, float64, one per species) are read before any cast to
    /// float32, and Σ_i E0(Z_i) over the N atoms, summed in f64 in atom order, is added to every
    /// bead's energy. An export without `interface_version()` (and without `atomic_energies()`)
    /// is v1: its energy is used as returned (P-6).
    ///
    /// Errors: missing or invalid file, or a module that is not an export -> `Backend`;
    /// `interface_version()` other than 2 (1 included), `atomic_energies()` without
    /// `interface_version()`, or a v2 export whose `atomic_energies()` is missing, not of the
    /// length of `atomic_numbers()` or not finite, or whose E0 sum over the atoms is not finite
    /// -> `Backend`;
    /// a float32 export with `Dtype::F64` -> `Backend` (Q27); a libtorch error (or panic) while
    /// casting the model to float32 -> `Backend`;
    /// an atomic number the model does not know -> `InvalidParameter { what: "atomic number" }`;
    /// `threads == 0` -> `InvalidParameter { what: "threads" }`; CUDA unavailable or a CUDA index
    /// at or above the device count -> `Backend`.
    pub fn load(
        path: &Path,
        atomic_numbers: &[u8],
        device: Device,
        dtype: Dtype,
        threads: usize,
        deterministic: bool,
    ) -> Result<MacePotential, PotentialError> {
        let n_threads = i32::try_from(threads).ok().filter(|&t| t >= 1).ok_or(
            PotentialError::InvalidParameter {
                what: "threads",
                value: threads as f64,
            },
        )?;
        let device = match device {
            Device::Cpu => tch::Device::Cpu,
            Device::Cuda(i) => {
                if !tch::Cuda::is_available() {
                    return Err(PotentialError::Backend(
                        "CUDA requested but libtorch sees no CUDA device".to_string(),
                    ));
                }
                let count = tch::Cuda::device_count();
                if i64::try_from(i).map_or(true, |i| i >= count) {
                    return Err(PotentialError::Backend(format!(
                        "CUDA device {i} requested, libtorch sees {count}"
                    )));
                }
                tch::Device::Cuda(i)
            }
        };
        if !path.is_file() {
            return Err(PotentialError::Backend(format!(
                "{}: not a file",
                path.display()
            )));
        }
        let mut module = CModule::load_on_device(path, device)
            .map_err(|e| backend(&format!("loading {}", path.display()), e))?;
        let not_export = |e| backend("not a pimd-torch MACE export (export_mace.py)", e);
        let r_max_angstrom = module
            .method_ts::<Tensor>("r_max", &[])
            .and_then(|t| t.f_double_value(&[]))
            .map_err(not_export)?;
        let z: Vec<i64> = module
            .method_ts::<Tensor>("atomic_numbers", &[])
            .and_then(|t| t.f_to(tch::Device::Cpu))
            .and_then(|t| Vec::<i64>::try_from(&t))
            .map_err(not_export)?;
        let code = match module.method_is::<IValue>("dtype_code", &[]) {
            Ok(IValue::Int(c)) => c,
            Ok(v) => {
                return Err(PotentialError::Backend(format!(
                    "not a pimd-torch MACE export: dtype_code() returned {v:?}"
                )));
            }
            Err(e) => return Err(not_export(e)),
        };
        if code == 32 && dtype == Dtype::F64 {
            return Err(PotentialError::Backend(
                "float32 export refused for Dtype::F64 (it would run with float32-rounded \
                 weights; export with --dtype float64)"
                    .to_string(),
            ));
        }
        if !(r_max_angstrom.is_finite() && r_max_angstrom > 0.0) {
            return Err(PotentialError::Backend(format!(
                "model r_max {r_max_angstrom} Å is not positive and finite"
            )));
        }
        let species = atomic_numbers
            .iter()
            .map(|&a| {
                z.iter()
                    .position(|&m| m == i64::from(a))
                    .map(|k| k as i64)
                    .ok_or(PotentialError::InvalidParameter {
                        what: "atomic number",
                        value: f64::from(a),
                    })
            })
            .collect::<Result<Vec<i64>, PotentialError>>()?;
        // Interface v2 (P-6): read the E0 table before any cast to float32 (which would round it).
        let e0_sum = match module.method_is::<IValue>("interface_version", &[]) {
            // No interface_version(): a v1 export, unless it has an E0 table (then it is neither).
            Err(_) => {
                if module.method_ts::<Tensor>("atomic_energies", &[]).is_ok() {
                    return Err(PotentialError::Backend(
                        "atomic_energies() without interface_version(): not a v1 or v2 export"
                            .to_string(),
                    ));
                }
                None
            }
            Ok(IValue::Int(2)) => {
                let e0: Vec<f64> = module
                    .method_ts::<Tensor>("atomic_energies", &[])
                    .and_then(|t| {
                        t.f_to_dtype_layout((Kind::Double, tch::Device::Cpu), false, false)
                    })
                    .and_then(|t| Vec::<f64>::try_from(&t))
                    .map_err(|e| backend("v2 export without a valid atomic_energies()", e))?;
                if e0.len() != z.len() {
                    return Err(PotentialError::Backend(format!(
                        "atomic_energies() has {} values for {} atomic numbers",
                        e0.len(),
                        z.len()
                    )));
                }
                if let Some(k) = e0.iter().position(|e| !e.is_finite()) {
                    return Err(PotentialError::Backend(format!(
                        "atomic_energies() value {} (Z = {}) is not finite",
                        e0[k], z[k]
                    )));
                }
                // f64, in atom order (D-3); eV → Eh through `units` (U-3).
                let mut sum_ev = 0.0;
                for &s in &species {
                    sum_ev += e0[s as usize];
                }
                if !sum_ev.is_finite() {
                    return Err(PotentialError::Backend(format!(
                        "sum of atomic_energies() over the atoms is not finite ({sum_ev} eV)"
                    )));
                }
                Some(sum_ev / HARTREE_IN_EV)
            }
            Ok(v) => {
                return Err(PotentialError::Backend(format!(
                    "unsupported export interface: interface_version() returned {v:?}"
                )));
            }
        };
        let kind = match dtype {
            Dtype::F64 => Kind::Double,
            Dtype::F32 => Kind::Float,
        };
        // A float32 export with F64 was refused above, so only F32 from a float64 export casts.
        if dtype == Dtype::F32 && code != 32 {
            // `CModule::to` panics on a libtorch error (e.g. out of GPU memory).
            catch_unwind(AssertUnwindSafe(|| module.to(device, kind, false))).map_err(|p| {
                let msg = p
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| p.downcast_ref::<&str>().copied())
                    .unwrap_or("panic");
                PotentialError::Backend(format!("casting the model to {kind:?}: {msg}"))
            })?;
        }
        module.f_set_eval().map_err(|e| backend("eval mode", e))?;
        // Process-wide libtorch settings (P-6, D-3), only after everything above is validated.
        tch::jit::f_set_profiling_mode(false)
            .map_err(|e| backend("disabling the profiling executor", e))?;
        tch::set_num_threads(n_threads);
        ffi::pimd_torch_set_deterministic(i32::from(deterministic));
        Ok(MacePotential {
            module,
            species,
            r_max: r_max_angstrom / BOHR_IN_ANGSTROM,
            device,
            kind,
            threads: n_threads,
            edges: Edges::default(),
            constants: Vec::new(),
            beads_per_graph: None,
            e0_sum,
        })
    }

    /// Evaluates the P beads in consecutive chunks of `c` beads (the last chunk may be smaller),
    /// each chunk one batched graph, chunks in ascending bead order (P-6). Without this call,
    /// one graph of all beads (as before).
    /// Errors: c == 0 -> InvalidParameter { what: "beads per graph", value: 0.0 }.
    pub fn with_beads_per_graph(self, c: usize) -> Result<MacePotential, PotentialError> {
        if c == 0 {
            return Err(PotentialError::InvalidParameter {
                what: "beads per graph",
                value: 0.0,
            });
        }
        Ok(MacePotential {
            beads_per_graph: Some(c),
            ..self
        })
    }

    /// The chunk size set by `with_beads_per_graph`, or `None` (all beads in one graph).
    pub fn beads_per_graph(&self) -> Option<usize> {
        self.beads_per_graph
    }

    /// The model cutoff r_max in bohr.
    pub fn cutoff(&self) -> f64 {
        self.r_max
    }

    /// One batched graph of the beads of `q` (`[b, N, 3]`, bohr), whose first bead is bead
    /// `first` of the call (for error messages): appends energies (Eh) and forces (Eh/bohr) of
    /// its b beads, in bead order, to `energies` and `forces`.
    fn evaluate_graph(
        &mut self,
        q: &Beads,
        cell: Option<&Cell>,
        first: usize,
        energies: &mut Vec<f64>,
        forces: &mut Vec<f64>,
    ) -> Result<(), PotentialError> {
        let (p, n) = (q.n_beads(), q.n_atoms());
        self.edges.build(q, cell, self.r_max);
        let pos: Vec<f64> = q.as_slice().iter().map(|x| x * BOHR_IN_ANGSTROM).collect();
        let (e_ev, f_ev) = self
            .forward(&pos, p, n)
            .map_err(|e| backend("MACE evaluation", e))?;
        if e_ev.len() != p || f_ev.len() != p * n * 3 {
            return Err(PotentialError::Backend(format!(
                "model returned {} energies and {} force components for {p} beads of {n} atoms",
                e_ev.len(),
                f_ev.len()
            )));
        }
        if let Some(j) = e_ev.iter().position(|e| !e.is_finite()) {
            return Err(PotentialError::Backend(format!(
                "bead {}: non-finite energy",
                first + j
            )));
        }
        if let Some(k) = f_ev.iter().position(|f| !f.is_finite()) {
            return Err(PotentialError::Backend(format!(
                "bead {}: non-finite force",
                first + k / (3 * n)
            )));
        }
        match self.e0_sum {
            None => energies.extend(e_ev.iter().map(|e| e / HARTREE_IN_EV)),
            Some(e0) => energies.extend(e_ev.iter().map(|e| e / HARTREE_IN_EV + e0)),
        }
        forces.extend(f_ev.iter().map(|f| f * BOHR_IN_ANGSTROM / HARTREE_IN_EV));
        Ok(())
    }

    /// Runs the batched graph: positions `[P·N·3]` (Å), the edges in `self.edges`, P graphs of
    /// N atoms. Returns energies `[P]` (eV) and forces `[P·N·3]` (eV/Å).
    fn forward(
        &mut self,
        pos_angstrom: &[f64],
        p: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), TchError> {
        let dev = self.device;
        let opts = (self.kind, dev);
        let k = match self.constants.iter().position(|c| c.p == p) {
            Some(k) => k,
            None => {
                if self.constants.len() >= 2 {
                    self.constants.remove(0);
                }
                self.constants
                    .push(Constants::new(&self.species, p, n, opts)?);
                self.constants.len() - 1
            }
        };
        let c = &self.constants[k];
        let pn = (p * n) as i64;
        let ne = (self.edges.index.len() / 2) as i64;
        let positions = Tensor::f_from_slice(pos_angstrom)?
            .f_reshape([pn, 3])?
            .f_to_dtype_layout(opts, false, false)?;
        let edge_index = Tensor::f_from_slice(&self.edges.index)?
            .f_reshape([2, ne])?
            .f_to(dev)?;
        let shifts = Tensor::f_from_slice(&self.edges.shift)?
            .f_reshape([ne, 3])?
            .f_to_dtype_layout(opts, false, false)?;
        let unit_shifts = Tensor::f_from_slice(&self.edges.unit)?
            .f_reshape([ne, 3])?
            .f_to_dtype_layout(opts, false, false)?;
        let out = self.module.forward_is(&[
            IValue::Tensor(positions),
            IValue::Tensor(c.species.shallow_clone()),
            IValue::Tensor(edge_index),
            IValue::Tensor(shifts),
            IValue::Tensor(c.ptr.shallow_clone()),
            IValue::Bool(false),
            IValue::Tensor(unit_shifts),
            IValue::Tensor(c.cell.shallow_clone()),
        ])?;
        let to_vec = |v: &IValue| -> Result<Vec<f64>, TchError> {
            match v {
                IValue::Tensor(t) => {
                    let t = t
                        .f_to_dtype_layout((Kind::Double, tch::Device::Cpu), false, false)?
                        .f_reshape([-1])?;
                    Vec::<f64>::try_from(&t)
                }
                other => Err(TchError::Convert(format!(
                    "expected a tensor, got {other:?}"
                ))),
            }
        };
        match &out {
            IValue::Tuple(v) if v.len() == 3 => Ok((to_vec(&v[0])?, to_vec(&v[1])?)),
            other => Err(TchError::Convert(format!(
                "expected (energy, forces, virial), got {other:?}"
            ))),
        }
    }
}

/// Input tensors that depend only on P and the atoms: species `[P·N]`, ptr `[P+1]` and the
/// (unread, virials off) cell `[P, 3, 3]`, on the model's device.
struct Constants {
    /// Bead count they were built for.
    p: usize,
    species: Tensor,
    ptr: Tensor,
    cell: Tensor,
}

impl Constants {
    fn new(
        species: &[i64],
        p: usize,
        n: usize,
        opts: (Kind, tch::Device),
    ) -> Result<Constants, TchError> {
        let sp: Vec<i64> = (0..p).flat_map(|_| species.iter().copied()).collect();
        let ptr: Vec<i64> = (0..=p).map(|b| (b * n) as i64).collect();
        Ok(Constants {
            p,
            species: Tensor::f_from_slice(&sp)?.f_to(opts.1)?,
            ptr: Tensor::f_from_slice(&ptr)?.f_to(opts.1)?,
            cell: Tensor::f_zeros([p as i64, 3, 3], opts)?,
        })
    }
}

/// Edges of one bead: atoms numbered within the batch (bead j holds jN … jN+N−1).
#[derive(Default)]
struct BeadEdges {
    /// Sender i of each edge.
    sender: Vec<i64>,
    /// Receiver j of each edge.
    receiver: Vec<i64>,
    /// Shift s per edge, Å, flat `[E_b, 3]` (edge vector q_j − q_i + s).
    shift: Vec<f64>,
    /// Shift in box lengths per edge (integers), flat `[E_b, 3]`.
    unit: Vec<f64>,
}

/// Edge list of the batch, beads in order 0 … P−1; the buffers are reused between calls.
#[derive(Default)]
struct Edges {
    /// Per-bead lists, built in parallel.
    beads: Vec<BeadEdges>,
    /// `[2, E]` flat: all senders, then all receivers.
    index: Vec<i64>,
    /// Shifts, Å, flat `[E, 3]`.
    shift: Vec<f64>,
    /// Shifts in box lengths, flat `[E, 3]`.
    unit: Vec<f64>,
}

impl Edges {
    /// P-6 edges: every ordered pair i ≠ j of the same bead with |q_j − q_i + s| < r_max, s the
    /// C-2 minimum-image correction with a cell, 0 without. All in bohr; shifts stored in Å.
    /// Beads are built in parallel and joined in bead order, so the result does not depend on
    /// the thread count (D-3).
    fn build(&mut self, q: &Beads, cell: Option<&Cell>, r_max: f64) {
        let p = q.n_beads();
        self.beads.resize_with(p, BeadEdges::default);
        self.beads
            .par_iter_mut()
            .enumerate()
            .for_each(|(b, e)| bead_edges(q, b, cell, r_max * r_max, e));
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

/// The edges of bead `b` into `e` (cleared first); `r2` = r_max², bohr².
fn bead_edges(q: &Beads, b: usize, cell: Option<&Cell>, r2: f64, e: &mut BeadEdges) {
    let n = q.n_atoms();
    let l = cell.map_or([0.0; 3], Cell::lengths);
    e.sender.clear();
    e.receiver.clear();
    e.shift.clear();
    e.unit.clear();
    for i in 0..n {
        let qi = q.get(b, i);
        for j in 0..n {
            if i == j {
                continue;
            }
            let qj = q.get(b, j);
            let d = [0, 1, 2].map(|k| qj[k] - qi[k]);
            let v = cell.map_or(d, |c| c.min_image(d));
            if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < r2 {
                // Shift in box lengths, computed only for kept edges (0 without a cell).
                let unit = match cell {
                    Some(_) => [0, 1, 2].map(|k| ((v[k] - d[k]) / l[k]).round()),
                    None => [0.0; 3],
                };
                e.sender.push((b * n + i) as i64);
                e.receiver.push((b * n + j) as i64);
                e.shift
                    .extend((0..3).map(|k| unit[k] * l[k] * BOHR_IN_ANGSTROM));
                e.unit.extend_from_slice(&unit);
            }
        }
    }
}

impl Potential for MacePotential {
    /// P-6: energies (Eh, per bead) and forces (Eh/bohr, `[P, N, 3]`) of all beads of
    /// `positions` (`[P, N, 3]`, bohr), in consecutive chunks of `beads_per_graph` beads (all
    /// beads by default), each chunk one batched graph; `virial` is `None`.
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        // libtorch keeps the intra-op thread count per OS thread (P-6, D-3).
        tch::set_num_threads(self.threads);
        if subset.is_some() {
            return Err(PotentialError::Unsupported);
        }
        let (p, n) = (positions.n_beads(), positions.n_atoms());
        if n != self.species.len() {
            return Err(PotentialError::ShapeMismatch {
                expected: self.species.len(),
                actual: n,
            });
        }
        if let Some(c) = cell
            && self.r_max >= 0.5 * c.min_length()
        {
            return Err(PotentialError::InvalidParameter {
                what: "cutoff",
                value: self.r_max,
            });
        }
        // P-6: consecutive chunks of c beads in ascending bead order, each one graph evaluated
        // exactly as a call with those beads alone; c ≥ P (or unset) is one graph of all beads.
        let c = self.beads_per_graph.map_or(p, |c| c.min(p)).max(1);
        let mut energies = Vec::with_capacity(p);
        let mut forces = Vec::with_capacity(p * n * 3);
        if c >= p {
            self.evaluate_graph(positions, cell, 0, &mut energies, &mut forces)?;
        } else {
            let n3 = n * 3;
            for s in (0..p).step_by(c) {
                let b = c.min(p - s);
                let chunk =
                    Beads::from_vec(b, n, positions.as_slice()[s * n3..(s + b) * n3].to_vec())
                        .map_err(|e| PotentialError::Backend(e.to_string()))?;
                self.evaluate_graph(&chunk, cell, s, &mut energies, &mut forces)?;
            }
        }
        Ok(Evaluation {
            energies,
            forces: Beads::from_vec(p, n, forces)
                .map_err(|e| PotentialError::Backend(e.to_string()))?,
            virial: None,
        })
    }
}
