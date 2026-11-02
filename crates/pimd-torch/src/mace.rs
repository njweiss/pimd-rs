//! `MacePotential` (SPEC P-6): a TorchScript MACE export (`scripts/export_mace.py`) evaluated
//! in-process, all P beads as one batched graph.

use std::path::Path;

use pimd_core::units::{BOHR_IN_ANGSTROM, HARTREE_IN_EV};
use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, Potential, PotentialError};
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
}

impl MacePotential {
    /// Loads a TorchScript export (export_mace.py) for N = `atomic_numbers.len()` atoms, sets the
    /// libtorch intra-op thread count to `threads`, disables the TorchScript profiling executor,
    /// and sets libtorch's deterministic-algorithms switch to `deterministic` (process-wide).
    ///
    /// Errors: missing or invalid file, or a module that is not an export -> `Backend`;
    /// an atomic number the model does not know -> `InvalidParameter { what: "atomic number" }`;
    /// `threads == 0` -> `InvalidParameter { what: "threads" }`; CUDA unavailable -> `Backend`.
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
        // Process-wide libtorch settings (P-6, D-3).
        tch::set_num_threads(n_threads);
        tch::jit::f_set_profiling_mode(false)
            .map_err(|e| backend("disabling the profiling executor", e))?;
        ffi::pimd_torch_set_deterministic(i32::from(deterministic));

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
        if !(r_max_angstrom.is_finite() && r_max_angstrom > 0.0) {
            return Err(PotentialError::Backend(format!(
                "model r_max {r_max_angstrom} Å is not positive and finite"
            )));
        }
        let species = atomic_numbers
            .iter()
            .map(|&a| {
                z.iter().position(|&m| m == i64::from(a)).map_or(
                    Err(PotentialError::InvalidParameter {
                        what: "atomic number",
                        value: f64::from(a),
                    }),
                    |k| Ok(k as i64),
                )
            })
            .collect::<Result<Vec<i64>, PotentialError>>()?;
        let kind = match dtype {
            Dtype::F64 => Kind::Double,
            Dtype::F32 => Kind::Float,
        };
        let export_kind = if code == 32 {
            Kind::Float
        } else {
            Kind::Double
        };
        if kind != export_kind {
            module.to(device, kind, false);
        }
        module.f_set_eval().map_err(|e| backend("eval mode", e))?;
        Ok(MacePotential {
            module,
            species,
            r_max: r_max_angstrom / BOHR_IN_ANGSTROM,
            device,
            kind,
        })
    }

    /// The model cutoff r_max in bohr.
    pub fn cutoff(&self) -> f64 {
        self.r_max
    }

    /// Runs the batched graph: positions `[P·N·3]` (Å), the edges, P graphs of N atoms.
    /// Returns energies `[P]` (eV) and forces `[P·N·3]` (eV/Å).
    fn forward(
        &self,
        pos_angstrom: &[f64],
        edges: &Edges,
        p: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), TchError> {
        let (dev, kind) = (self.device, self.kind);
        let opts = (kind, dev);
        let to_i64 = |x: usize| x as i64;
        let pn = to_i64(p * n);
        let ne = to_i64(edges.sender.len());
        let positions = Tensor::f_from_slice(pos_angstrom)?
            .f_reshape([pn, 3])?
            .f_to_dtype_layout(opts, false, false)?;
        let sp: Vec<i64> = (0..p).flat_map(|_| self.species.iter().copied()).collect();
        let species = Tensor::f_from_slice(&sp)?.f_to(dev)?;
        let mut ei = edges.sender.clone();
        ei.extend_from_slice(&edges.receiver);
        let edge_index = Tensor::f_from_slice(&ei)?.f_reshape([2, ne])?.f_to(dev)?;
        let shifts = Tensor::f_from_slice(&edges.shift)?
            .f_reshape([ne, 3])?
            .f_to_dtype_layout(opts, false, false)?;
        let unit_shifts = Tensor::f_from_slice(&edges.unit)?
            .f_reshape([ne, 3])?
            .f_to_dtype_layout(opts, false, false)?;
        let ptr: Vec<i64> = (0..=p).map(|b| to_i64(b * n)).collect();
        let ptr = Tensor::f_from_slice(&ptr)?.f_to(dev)?;
        let cell = Tensor::f_zeros([to_i64(p), 3, 3], opts)?;
        let out = self.module.forward_is(&[
            IValue::Tensor(positions),
            IValue::Tensor(species),
            IValue::Tensor(edge_index),
            IValue::Tensor(shifts),
            IValue::Tensor(ptr),
            IValue::Bool(false),
            IValue::Tensor(unit_shifts),
            IValue::Tensor(cell),
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

/// Edge list of the batch: atoms numbered within the batch (bead j holds jN … jN+N−1).
#[derive(Default)]
struct Edges {
    /// Sender i of each edge.
    sender: Vec<i64>,
    /// Receiver j of each edge.
    receiver: Vec<i64>,
    /// Shift s per edge, Å, flat `[E, 3]` (edge vector q_j − q_i + s).
    shift: Vec<f64>,
    /// Shift in box lengths per edge (integers), flat `[E, 3]`.
    unit: Vec<f64>,
}

/// P-6 edges: every ordered pair i ≠ j of the same bead with |q_j − q_i + s| < r_max, s the C-2
/// minimum-image correction with a cell, 0 without. All in bohr; shifts returned in Å.
fn build_edges(q: &Beads, cell: Option<&Cell>, r_max: f64) -> Edges {
    let (p, n) = (q.n_beads(), q.n_atoms());
    let r2 = r_max * r_max;
    let lengths = cell.map(Cell::lengths);
    let mut e = Edges::default();
    for b in 0..p {
        for i in 0..n {
            let qi = q.get(b, i);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let qj = q.get(b, j);
                let d = [0, 1, 2].map(|k| qj[k] - qi[k]);
                let (v, unit) = match (cell, lengths) {
                    (Some(c), Some(l)) => {
                        let v = c.min_image(d);
                        (v, [0, 1, 2].map(|k| ((v[k] - d[k]) / l[k]).round()))
                    }
                    _ => (d, [0.0; 3]),
                };
                if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < r2 {
                    e.sender.push((b * n + i) as i64);
                    e.receiver.push((b * n + j) as i64);
                    let l = lengths.unwrap_or([0.0; 3]);
                    e.shift
                        .extend((0..3).map(|k| unit[k] * l[k] * BOHR_IN_ANGSTROM));
                    e.unit.extend_from_slice(&unit);
                }
            }
        }
    }
    e
}

impl Potential for MacePotential {
    /// P-6: energies (Eh, per bead) and forces (Eh/bohr, `[P, N, 3]`) of all beads of
    /// `positions` (`[P, N, 3]`, bohr) as one batched graph; `virial` is `None`.
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
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
        let edges = build_edges(positions, cell, self.r_max);
        let pos: Vec<f64> = positions
            .as_slice()
            .iter()
            .map(|x| x * BOHR_IN_ANGSTROM)
            .collect();
        let (e_ev, f_ev) = self
            .forward(&pos, &edges, p, n)
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
                "bead {j}: non-finite energy"
            )));
        }
        if let Some(k) = f_ev.iter().position(|f| !f.is_finite()) {
            return Err(PotentialError::Backend(format!(
                "bead {}: non-finite force",
                k / (3 * n)
            )));
        }
        let energies = e_ev.iter().map(|e| e / HARTREE_IN_EV).collect();
        let forces = f_ev
            .iter()
            .map(|f| f * BOHR_IN_ANGSTROM / HARTREE_IN_EV)
            .collect();
        Ok(Evaluation {
            energies,
            forces: Beads::from_vec(p, n, forces)
                .map_err(|e| PotentialError::Backend(e.to_string()))?,
            virial: None,
        })
    }
}
