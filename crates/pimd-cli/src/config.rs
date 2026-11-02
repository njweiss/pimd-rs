//! Run configuration (MILESTONE_4 "Config format"): TOML in user units (Å, K, amu, fs), stored
//! in Hartree atomic units (U-1). The conversion happens here only, through `pimd_core::units`
//! (U-3).

use pimd_core::units::{
    AMU_IN_ELECTRON_MASSES, BOHR_IN_ANGSTROM, FS_IN_ATOMIC_TIME, K_B_HARTREE_PER_KELVIN,
    beta_from_kelvin,
};
use pimd_core::{Cell, Transform};
use pimd_potentials::LjParams;
use pimd_socket::Address;
use serde::Deserialize;

use crate::error::CliError;

/// A validated run configuration in atomic units. Built by `Config::from_toml_str`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `[system]`.
    pub system: SystemConfig,
    /// `[potential]`.
    pub potential: PotentialConfig,
    /// `[sampling]`.
    pub sampling: SamplingConfig,
    /// `[output]`.
    pub output: OutputConfig,
}

/// `[system]`: N atoms with per-atom masses, in an orthorhombic periodic cell or in the gas phase
/// (MILESTONE_5 "Config format").
#[derive(Debug, Clone, PartialEq)]
pub struct SystemConfig {
    /// Number of atoms N.
    pub n_atoms: usize,
    /// Mass of each atom, length N, m_e (from `masses_amu`, or `mass_amu` for every atom).
    pub masses: Vec<f64>,
    /// Periodic cell, bohr (from `box_angstrom`, one value for a cube or three); `None` = gas
    /// phase (C-1).
    pub cell: Option<Cell>,
    /// Initial position of each atom, length N, bohr (from `positions_angstrom`); `None` = the
    /// M4 lattice (requires a cell).
    pub positions: Option<Vec<[f64; 3]>>,
    /// Element symbol of each atom, length N, each non-empty without whitespace, for output only
    /// (`symbols`).
    pub symbols: Option<Vec<String>>,
}

/// `[potential]`.
#[derive(Debug, Clone, PartialEq)]
pub enum PotentialConfig {
    /// `kind = "lj"`: ε in Eh, σ, cutoff and skin in bohr, shift flag (P-3).
    Lj(LjParams),
    /// `kind = "socket"`: an i-PI protocol server (K-1) at `address` serving `n_clients ≥ 1`
    /// clients, with `0 < timeout_s ≤ 1e9` seconds for accepting them and for every read and write.
    Socket {
        /// Where the server listens.
        address: Address,
        /// Number of clients to wait for.
        n_clients: usize,
        /// Accept, read and write timeout, s, in (0, 1e9].
        timeout_s: f64,
    },
    /// `kind = "mace"` (feature `torch`): a TorchScript MACE export evaluated in-process (P-6),
    /// for the atoms of `system.symbols` (required with this kind).
    #[cfg(feature = "torch")]
    Mace {
        /// Path of the TorchScript export (`export_mace.py`), as given.
        model: std::path::PathBuf,
        /// `device`: "cpu" (default), "cuda" (= CUDA device 0) or "cuda:N".
        device: pimd_torch::Device,
        /// `dtype`: "float64" (default) or "float32".
        dtype: pimd_torch::Dtype,
        /// `threads`: libtorch intra-op threads, ≥ 1 (default 1).
        threads: usize,
        /// `deterministic`: libtorch deterministic algorithms (default true).
        deterministic: bool,
        /// 64-bit FNV-1a of the model file's bytes, read at parse time (config hash; not the
        /// path).
        model_fnv1a: u64,
    },
}

/// `[sampling]`: PIMD with BAOAB + PILE-L (I-1, T-1).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    /// Temperature T, K, as given.
    pub temperature_kelvin: f64,
    /// β = 1/(k_B T), Eh⁻¹ (U-4).
    pub beta: f64,
    /// Number of beads P ≥ 1.
    pub n_beads: usize,
    /// Time step, ħ/Eh (from `dt_fs`).
    pub dt: f64,
    /// Centroid thermostat time constant τ_0, ħ/Eh (from `tau0_fs`, T-2).
    pub tau0: f64,
    /// Equilibration steps (nothing recorded).
    pub n_equil: usize,
    /// Production steps; step s = 1 … n_steps is recorded when s % stride == 0.
    pub n_steps: usize,
    /// Recording stride ≥ 1, in steps.
    pub stride: usize,
    /// Seed of the single `pimd_core::Rng` (D-2).
    pub seed: u64,
    /// Normal-mode transform algorithm (`normal_modes`, default "matrix", RP-4).
    pub normal_modes: Transform,
}

/// `[output]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputConfig {
    /// RDF settings when `rdf_bins` and `rdf_rmax_angstrom` are both given, else `None`.
    pub rdf: Option<RdfConfig>,
    /// Write `checkpoint.bin` every this many production steps (and always at the end).
    pub checkpoint_every: usize,
    /// Write a `trajectory.xyz` frame at production steps s with s % trajectory_every == 0.
    pub trajectory_every: Option<usize>,
}

/// RDF histogram settings (E-7).
#[derive(Debug, Clone, PartialEq)]
pub struct RdfConfig {
    /// Number of bins.
    pub n_bins: usize,
    /// Upper edge r_max, bohr (from `rdf_rmax_angstrom`), ≤ L_min/2.
    pub r_max: f64,
}

impl Config {
    /// Parses and validates a TOML config, converting user units to atomic units (U-3).
    ///
    /// Errors: `CliError::Parse` for TOML syntax errors, missing fields, unknown keys and
    /// wrongly typed values; `CliError::InvalidValue` (naming the field) for out-of-range values,
    /// including `n_beads = 0` and `cutoff + skin > L_min/2` (C-4); `CliError::Io` (naming
    /// `potential.model` and the path) for a model file that cannot be read (kind = "mace": the
    /// file is read and hashed at parse time, Q24).
    pub fn from_toml_str(s: &str) -> Result<Config, CliError> {
        let raw: RawConfig = toml::from_str(s).map_err(|e| CliError::Parse(e.to_string()))?;
        raw.validate()
    }

    /// 64-bit FNV-1a over a canonical little-endian encoding of the parsed `[system]`,
    /// `[potential]` and `[sampling]` (atomic units), excluding `n_steps`, `symbols` (except for
    /// kind = "mace") and the socket's `address`, `n_clients` and `timeout_s`; `[output]` is not
    /// covered (MILESTONE_4 "Restart determinism", MILESTONE_5 "Config hash"). For kind = "mace"
    /// it covers the model file's bytes (not its path), the device kind (not the index), dtype,
    /// threads, deterministic and the symbols (MILESTONE_6 "Config"). A checkpoint restarts only
    /// under a config with the same hash.
    pub(crate) fn hash(&self) -> u64 {
        let (sy, sa) = (&self.system, &self.sampling);
        let mut bytes = Vec::new();
        let mut u = |v: u64| put_u64(&mut bytes, v);
        u(sy.n_atoms as u64);
        for m in &sy.masses {
            u(m.to_bits());
        }
        match &sy.cell {
            None => u(0),
            Some(cell) => {
                u(1);
                for l in cell.lengths() {
                    u(l.to_bits());
                }
            }
        }
        match &sy.positions {
            None => u(0),
            Some(x) => {
                u(1);
                for v in x.iter().flatten() {
                    u(v.to_bits());
                }
            }
        }
        // `symbols` and the socket's connection details (address, n_clients, timeout_s) do not
        // change results and are excluded (MILESTONE_5 "Config hash").
        match &self.potential {
            PotentialConfig::Lj(po) => {
                u(0);
                u(po.epsilon.to_bits());
                u(po.sigma.to_bits());
                u(po.cutoff.to_bits());
                u(po.skin.to_bits());
                u(u64::from(po.shift));
            }
            PotentialConfig::Socket { .. } => u(1),
            #[cfg(feature = "torch")]
            PotentialConfig::Mace {
                device,
                dtype,
                threads,
                deterministic,
                model_fnv1a,
                ..
            } => {
                u(2);
                u(*model_fnv1a);
                u(match device {
                    pimd_torch::Device::Cpu => 0,
                    pimd_torch::Device::Cuda(_) => 1,
                });
                u(match dtype {
                    pimd_torch::Dtype::F64 => 0,
                    pimd_torch::Dtype::F32 => 1,
                });
                u(*threads as u64);
                u(u64::from(*deterministic));
                // The symbols select the model's species (unlike for the socket).
                // Hashed as atomic numbers (validated at parse time; 0 for anything else).
                for s in sy.symbols.iter().flatten() {
                    u(pimd_torch::atomic_number(s).map_or(0, u64::from));
                }
            }
        }
        u(sa.temperature_kelvin.to_bits());
        u(sa.beta.to_bits());
        u(sa.n_beads as u64);
        u(sa.dt.to_bits());
        u(sa.tau0.to_bits());
        u(sa.n_equil as u64);
        u(sa.stride as u64);
        u(sa.seed);
        u(match sa.normal_modes {
            Transform::Matrix => 0,
            Transform::Fft => 1,
        });
        fnv1a(&bytes)
    }
}

/// 64-bit FNV-1a.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The TOML file as written, in user units. Unknown keys and sections are rejected.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    system: RawSystem,
    potential: RawPotential,
    sampling: RawSampling,
    output: RawOutput,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSystem {
    n_atoms: usize,
    mass_amu: Option<f64>,
    masses_amu: Option<Vec<f64>>,
    symbols: Option<Vec<String>>,
    positions_angstrom: Option<Vec<[f64; 3]>>,
    box_angstrom: Option<RawBox>,
}

/// `box_angstrom`: one edge (cube) or a list of edges (three required).
#[derive(Deserialize)]
#[serde(untagged)]
enum RawBox {
    Cube(f64),
    Edges(Vec<f64>),
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawKind {
    Lj,
    Socket,
    Mace,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPotential {
    kind: RawKind,
    epsilon_kelvin: Option<f64>,
    sigma_angstrom: Option<f64>,
    cutoff_sigma: Option<f64>,
    skin_sigma: Option<f64>,
    shift: Option<bool>,
    address: Option<String>,
    n_clients: Option<usize>,
    timeout_s: Option<f64>,
    model: Option<String>,
    device: Option<String>,
    dtype: Option<String>,
    threads: Option<usize>,
    deterministic: Option<bool>,
}

impl RawPotential {
    /// `Parse` naming the first of the keys of another kind that are present, if any.
    fn reject_keys(&self, kind: &str, keys: &[(&str, bool)]) -> Result<(), CliError> {
        match keys.iter().find(|(_, present)| *present) {
            Some((key, _)) => Err(CliError::Parse(format!(
                "potential: `{key}` does not belong to kind = \"{kind}\""
            ))),
            None => Ok(()),
        }
    }

    /// The Lennard-Jones keys and whether each is present.
    fn lj_keys(&self) -> [(&'static str, bool); 5] {
        [
            ("epsilon_kelvin", self.epsilon_kelvin.is_some()),
            ("sigma_angstrom", self.sigma_angstrom.is_some()),
            ("cutoff_sigma", self.cutoff_sigma.is_some()),
            ("skin_sigma", self.skin_sigma.is_some()),
            ("shift", self.shift.is_some()),
        ]
    }

    /// The socket keys and whether each is present.
    fn socket_keys(&self) -> [(&'static str, bool); 3] {
        [
            ("address", self.address.is_some()),
            ("n_clients", self.n_clients.is_some()),
            ("timeout_s", self.timeout_s.is_some()),
        ]
    }

    /// The MACE keys and whether each is present.
    fn mace_keys(&self) -> [(&'static str, bool); 5] {
        [
            ("model", self.model.is_some()),
            ("device", self.device.is_some()),
            ("dtype", self.dtype.is_some()),
            ("threads", self.threads.is_some()),
            ("deterministic", self.deterministic.is_some()),
        ]
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum RawNormalModes {
    #[default]
    Matrix,
    Fft,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSampling {
    temperature_kelvin: f64,
    n_beads: usize,
    dt_fs: f64,
    tau0_fs: f64,
    n_equil: usize,
    n_steps: usize,
    stride: usize,
    seed: u64,
    #[serde(default)]
    normal_modes: RawNormalModes,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutput {
    rdf_bins: Option<usize>,
    rdf_rmax_angstrom: Option<f64>,
    checkpoint_every: usize,
    trajectory_every: Option<usize>,
}

/// Appends `v` as 8 little-endian bytes (config hash and checkpoint encoding).
pub(crate) fn put_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}

/// C-1: the gas phase has no lattice to start from.
pub(crate) const GAS_PHASE_NEEDS_POSITIONS: &str =
    "system: `positions_angstrom` is required without `box_angstrom` (gas phase)";

/// `InvalidValue` naming `field` (`section.key`).
pub(crate) fn invalid(field: &str, value: f64) -> CliError {
    CliError::InvalidValue {
        what: field.to_string(),
        value,
    }
}

/// `v` if positive and finite, else `InvalidValue` naming `field`.
fn positive(field: &str, v: f64) -> Result<f64, CliError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(invalid(field, v))
    }
}

/// Whether `n` f64 values fit in one allocation (`n · 8` bytes at most `isize::MAX`), with
/// checked arithmetic (Q17).
fn fits_f64s(n: Option<usize>) -> bool {
    n.and_then(|n| n.checked_mul(8))
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

/// `v` if present, else `Parse` naming the missing `key`.
pub(crate) fn need<T>(v: Option<T>, key: &str) -> Result<T, CliError> {
    v.ok_or_else(|| CliError::Parse(format!("missing field `{key}`")))
}

/// `n` if at least 1, else `InvalidValue` naming `field`.
fn at_least_one(field: &str, n: usize) -> Result<usize, CliError> {
    if n >= 1 {
        Ok(n)
    } else {
        Err(invalid(field, 0.0))
    }
}

impl RawConfig {
    /// Range checks and the conversion to atomic units (U-3), only through `pimd_core::units`.
    fn validate(self) -> Result<Config, CliError> {
        let RawConfig {
            system: sy,
            potential: po,
            sampling: sa,
            output: out,
        } = self;

        let n_atoms = at_least_one("system.n_atoms", sy.n_atoms)?;
        let n_beads = at_least_one("sampling.n_beads", sa.n_beads)?;
        // Q17: sizes that cannot be allocated, checked before anything of that size is.
        if !fits_f64s(n_atoms.checked_mul(3)) {
            return Err(invalid("system.n_atoms", n_atoms as f64));
        }
        if !fits_f64s(n_beads.checked_mul(n_beads)) {
            return Err(invalid("sampling.n_beads", n_beads as f64));
        }
        if !fits_f64s(n_beads.checked_mul(n_atoms).and_then(|n| n.checked_mul(3))) {
            return Err(invalid("system.n_atoms", n_atoms as f64));
        }
        let masses = match (sy.mass_amu, sy.masses_amu) {
            (Some(m), None) => {
                vec![positive("system.mass_amu", m)? * AMU_IN_ELECTRON_MASSES; n_atoms]
            }
            (None, Some(v)) => {
                if v.len() != n_atoms {
                    return Err(invalid("system.masses_amu", v.len() as f64));
                }
                v.into_iter()
                    .map(|m| Ok(positive("system.masses_amu", m)? * AMU_IN_ELECTRON_MASSES))
                    .collect::<Result<Vec<f64>, CliError>>()?
            }
            _ => {
                return Err(CliError::Parse(
                    "system: exactly one of `mass_amu` and `masses_amu` must be given".to_string(),
                ));
            }
        };
        if let Some(v) = &sy.symbols {
            if v.len() != n_atoms {
                return Err(invalid("system.symbols", v.len() as f64));
            }
            if let Some(i) = v
                .iter()
                .position(|s| s.is_empty() || s.chars().any(char::is_whitespace))
            {
                return Err(invalid("system.symbols", i as f64));
            }
        }
        let positions = match sy.positions_angstrom {
            None => None,
            Some(v) => {
                if v.len() != n_atoms {
                    return Err(invalid("system.positions_angstrom", v.len() as f64));
                }
                let mut out = Vec::with_capacity(n_atoms);
                for x in v {
                    if let Some(&bad) = x.iter().find(|c| !c.is_finite()) {
                        return Err(invalid("system.positions_angstrom", bad));
                    }
                    out.push(x.map(|c| c / BOHR_IN_ANGSTROM));
                }
                Some(out)
            }
        };
        let cell = match sy.box_angstrom {
            None => None,
            Some(box_angstrom) => {
                let edges = match box_angstrom {
                    RawBox::Cube(l) => [l; 3],
                    RawBox::Edges(v) => <[f64; 3]>::try_from(v.as_slice()).map_err(|_| {
                        CliError::Parse(format!(
                            "system.box_angstrom: expected one value or three, got {}",
                            v.len()
                        ))
                    })?,
                };
                let mut lengths = [0.0; 3];
                for (l, &a) in lengths.iter_mut().zip(&edges) {
                    *l = positive("system.box_angstrom", a)? / BOHR_IN_ANGSTROM;
                }
                Some(Cell::new(lengths)?)
            }
        };
        if cell.is_none() && positions.is_none() {
            return Err(CliError::Parse(GAS_PHASE_NEEDS_POSITIONS.to_string()));
        }

        let potential = match po.kind {
            RawKind::Lj => {
                po.reject_keys("lj", &po.mace_keys())?;
                if po.socket_keys().iter().any(|&(_, present)| present) {
                    return Err(CliError::Parse(
                        "potential: `address`, `n_clients` and `timeout_s` belong to kind = \"socket\""
                            .to_string(),
                    ));
                }
                let epsilon_kelvin = need(po.epsilon_kelvin, "epsilon_kelvin")?;
                let sigma_angstrom = need(po.sigma_angstrom, "sigma_angstrom")?;
                let cutoff_sigma = need(po.cutoff_sigma, "cutoff_sigma")?;
                let skin_sigma = need(po.skin_sigma, "skin_sigma")?;
                let shift = need(po.shift, "shift")?;
                let epsilon =
                    positive("potential.epsilon_kelvin", epsilon_kelvin)? * K_B_HARTREE_PER_KELVIN;
                let sigma =
                    positive("potential.sigma_angstrom", sigma_angstrom)? / BOHR_IN_ANGSTROM;
                let cutoff_sigma = positive("potential.cutoff_sigma", cutoff_sigma)?;
                if !(skin_sigma.is_finite() && skin_sigma >= 0.0) {
                    return Err(invalid("potential.skin_sigma", skin_sigma));
                }
                let lj = LjParams {
                    epsilon,
                    sigma,
                    cutoff: cutoff_sigma * sigma,
                    skin: skin_sigma * sigma,
                    shift,
                };
                // C-4: cutoff + skin ≤ L_min/2 (periodic cells only).
                if let Some(cell) = &cell
                    && lj.cutoff + lj.skin > 0.5 * cell.min_length()
                {
                    return Err(invalid("potential.cutoff_sigma", cutoff_sigma + skin_sigma));
                }
                PotentialConfig::Lj(lj)
            }
            RawKind::Socket => {
                po.reject_keys("socket", &po.mace_keys())?;
                if po.lj_keys().iter().any(|&(_, present)| present) {
                    return Err(CliError::Parse(
                        "potential: Lennard-Jones parameters belong to kind = \"lj\"".to_string(),
                    ));
                }
                let address = need(po.address, "address")?
                    .parse::<Address>()
                    .map_err(|e| CliError::Parse(format!("potential.address: {e}")))?;
                let n_clients = need(po.n_clients, "n_clients")?;
                let timeout_s = positive("potential.timeout_s", need(po.timeout_s, "timeout_s")?)?;
                if timeout_s > 1e9 {
                    return Err(invalid("potential.timeout_s", timeout_s));
                }
                PotentialConfig::Socket {
                    address,
                    n_clients: at_least_one("potential.n_clients", n_clients)?,
                    timeout_s,
                }
            }
            RawKind::Mace => mace(&po, sy.symbols.as_deref())?,
        };

        let temperature_kelvin = positive("sampling.temperature_kelvin", sa.temperature_kelvin)?;
        let sampling = SamplingConfig {
            temperature_kelvin,
            beta: beta_from_kelvin(temperature_kelvin)?,
            n_beads,
            dt: positive("sampling.dt_fs", sa.dt_fs)? * FS_IN_ATOMIC_TIME,
            tau0: positive("sampling.tau0_fs", sa.tau0_fs)? * FS_IN_ATOMIC_TIME,
            n_equil: sa.n_equil,
            n_steps: sa.n_steps,
            stride: at_least_one("sampling.stride", sa.stride)?,
            seed: sa.seed,
            normal_modes: match sa.normal_modes {
                RawNormalModes::Matrix => Transform::Matrix,
                RawNormalModes::Fft => Transform::Fft,
            },
        };

        let rdf = match (out.rdf_bins, out.rdf_rmax_angstrom) {
            (None, None) => None,
            (Some(n_bins), Some(r)) => {
                let n_bins = at_least_one("output.rdf_bins", n_bins)?;
                let r_max = positive("output.rdf_rmax_angstrom", r)? / BOHR_IN_ANGSTROM;
                // E-7 needs a periodic cell.
                let Some(cell) = &cell else {
                    return Err(CliError::Parse(
                        "output: rdf_bins and rdf_rmax_angstrom require system.box_angstrom"
                            .to_string(),
                    ));
                };
                // E-7: r_max ≤ L_min/2.
                if r_max > 0.5 * cell.min_length() {
                    return Err(invalid("output.rdf_rmax_angstrom", r));
                }
                Some(RdfConfig { n_bins, r_max })
            }
            _ => {
                return Err(CliError::Parse(
                    "output: rdf_bins and rdf_rmax_angstrom must be given together".to_string(),
                ));
            }
        };
        // E-7 needs N ≥ 2 (N2).
        if rdf.is_some() && n_atoms < 2 {
            return Err(invalid("system.n_atoms", n_atoms as f64));
        }
        let output = OutputConfig {
            rdf,
            checkpoint_every: at_least_one("output.checkpoint_every", out.checkpoint_every)?,
            trajectory_every: out
                .trajectory_every
                .map(|t| at_least_one("output.trajectory_every", t))
                .transpose()?,
        };

        Ok(Config {
            system: SystemConfig {
                n_atoms,
                masses,
                cell,
                positions,
                symbols: sy.symbols,
            },
            potential,
            sampling,
            output,
        })
    }
}

/// `kind = "mace"` without the feature `torch`: a config error naming the feature.
#[cfg(not(feature = "torch"))]
fn mace(_po: &RawPotential, _symbols: Option<&[String]>) -> Result<PotentialConfig, CliError> {
    Err(CliError::Parse(
        "potential: kind = \"mace\" needs pimd-rs built with the feature `torch`".to_string(),
    ))
}

/// `kind = "mace"` (P-6, MILESTONE_6 "Config"): `model` required, `device`, `dtype`, `threads`,
/// `deterministic` with their defaults; `system.symbols` required and known to
/// `pimd_torch::atomic_number`. Reads the model file to hash its bytes.
#[cfg(feature = "torch")]
fn mace(po: &RawPotential, symbols: Option<&[String]>) -> Result<PotentialConfig, CliError> {
    po.reject_keys("mace", &po.lj_keys())?;
    po.reject_keys("mace", &po.socket_keys())?;
    let symbols = need(symbols, "symbols")?;
    if let Some(i) = symbols
        .iter()
        .position(|s| pimd_torch::atomic_number(s).is_none())
    {
        return Err(invalid("system.symbols", i as f64));
    }
    let model = std::path::PathBuf::from(need(po.model.as_ref(), "model")?);
    let device = match po.device.as_deref() {
        None | Some("cpu") => pimd_torch::Device::Cpu,
        Some("cuda") => pimd_torch::Device::Cuda(0),
        Some(s) => {
            let index = s
                .strip_prefix("cuda:")
                .filter(|i| !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|i| i.parse::<usize>().ok())
                .ok_or_else(|| {
                    CliError::Parse(format!(
                        "potential.device: expected \"cpu\", \"cuda\" or \"cuda:N\", got {s:?}"
                    ))
                })?;
            pimd_torch::Device::Cuda(index)
        }
    };
    let dtype = match po.dtype.as_deref() {
        None | Some("float64") => pimd_torch::Dtype::F64,
        Some("float32") => pimd_torch::Dtype::F32,
        Some(s) => {
            return Err(CliError::Parse(format!(
                "potential.dtype: expected \"float64\" or \"float32\", got {s:?}"
            )));
        }
    };
    let threads = at_least_one("potential.threads", po.threads.unwrap_or(1))?;
    let bytes = std::fs::read(&model).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("potential.model {}: {e}", model.display()),
        )
    })?;
    let model_fnv1a = fnv1a(&bytes);
    Ok(PotentialConfig::Mace {
        model,
        device,
        dtype,
        threads,
        deterministic: po.deterministic.unwrap_or(true),
        model_fnv1a,
    })
}
