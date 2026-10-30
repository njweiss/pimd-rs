//! Run configuration (MILESTONE_4 "Config format"): TOML in user units (Å, K, amu, fs), stored
//! in Hartree atomic units (U-1). The conversion happens here only, through `pimd_core::units`
//! (U-3).

use pimd_core::units::{
    AMU_IN_ELECTRON_MASSES, BOHR_IN_ANGSTROM, FS_IN_ATOMIC_TIME, K_B_HARTREE_PER_KELVIN,
    beta_from_kelvin,
};
use pimd_core::{Cell, Transform};
use pimd_potentials::LjParams;
use serde::Deserialize;

use crate::error::CliError;

/// A validated run configuration in atomic units. Built by `Config::from_toml_str`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `[system]`.
    pub system: SystemConfig,
    /// `[potential]` (`kind = "lj"`): ε in Eh, σ, cutoff and skin in bohr, shift flag (P-3).
    pub potential: LjParams,
    /// `[sampling]`.
    pub sampling: SamplingConfig,
    /// `[output]`.
    pub output: OutputConfig,
}

/// `[system]`: one species in an orthorhombic periodic cell.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemConfig {
    /// Number of atoms N.
    pub n_atoms: usize,
    /// Mass of every atom, m_e (from `mass_amu`).
    pub mass: f64,
    /// Periodic cell, bohr (from `box_angstrom`, one value for a cube or three).
    pub cell: Cell,
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
    /// including `n_beads = 0` and `cutoff + skin > L_min/2` (C-4).
    pub fn from_toml_str(s: &str) -> Result<Config, CliError> {
        let raw: RawConfig = toml::from_str(s).map_err(|e| CliError::Parse(e.to_string()))?;
        raw.validate()
    }

    /// 64-bit FNV-1a over a canonical little-endian encoding of the parsed `[system]`,
    /// `[potential]` and `[sampling]` (atomic units), excluding `n_steps`; `[output]` is not
    /// covered (MILESTONE_4 "Restart determinism"). A checkpoint restarts only under a config
    /// with the same hash.
    pub(crate) fn hash(&self) -> u64 {
        let (sy, po, sa) = (&self.system, &self.potential, &self.sampling);
        let mut bytes = Vec::new();
        let mut u = |v: u64| put_u64(&mut bytes, v);
        u(sy.n_atoms as u64);
        u(sy.mass.to_bits());
        for l in sy.cell.lengths() {
            u(l.to_bits());
        }
        u(po.epsilon.to_bits());
        u(po.sigma.to_bits());
        u(po.cutoff.to_bits());
        u(po.skin.to_bits());
        u(u64::from(po.shift));
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
    mass_amu: f64,
    box_angstrom: RawBox,
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
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPotential {
    #[allow(dead_code)] // Only "lj" exists; parsing it is the check.
    kind: RawKind,
    epsilon_kelvin: f64,
    sigma_angstrom: f64,
    cutoff_sigma: f64,
    skin_sigma: f64,
    shift: bool,
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
}

/// `InvalidValue` naming `field` (`section.key`).
/// Appends `v` as 8 little-endian bytes (config hash and checkpoint encoding).
pub(crate) fn put_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn invalid(field: &str, value: f64) -> CliError {
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
        let mass = positive("system.mass_amu", sy.mass_amu)? * AMU_IN_ELECTRON_MASSES;
        let edges = match sy.box_angstrom {
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
        let cell = Cell::new(lengths)?;

        let epsilon =
            positive("potential.epsilon_kelvin", po.epsilon_kelvin)? * K_B_HARTREE_PER_KELVIN;
        let sigma = positive("potential.sigma_angstrom", po.sigma_angstrom)? / BOHR_IN_ANGSTROM;
        let cutoff_sigma = positive("potential.cutoff_sigma", po.cutoff_sigma)?;
        if !(po.skin_sigma.is_finite() && po.skin_sigma >= 0.0) {
            return Err(invalid("potential.skin_sigma", po.skin_sigma));
        }
        let potential = LjParams {
            epsilon,
            sigma,
            cutoff: cutoff_sigma * sigma,
            skin: po.skin_sigma * sigma,
            shift: po.shift,
        };
        // C-4: cutoff + skin ≤ L_min/2.
        if potential.cutoff + potential.skin > 0.5 * cell.min_length() {
            return Err(invalid(
                "potential.cutoff_sigma",
                po.cutoff_sigma + po.skin_sigma,
            ));
        }

        let temperature_kelvin = positive("sampling.temperature_kelvin", sa.temperature_kelvin)?;
        let sampling = SamplingConfig {
            temperature_kelvin,
            beta: beta_from_kelvin(temperature_kelvin)?,
            n_beads: at_least_one("sampling.n_beads", sa.n_beads)?,
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
        };

        Ok(Config {
            system: SystemConfig {
                n_atoms,
                mass,
                cell,
            },
            potential,
            sampling,
            output,
        })
    }
}
