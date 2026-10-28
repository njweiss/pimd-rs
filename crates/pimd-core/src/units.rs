//! Physical constants and conversions (SPEC U-2, U-3). CODATA 2018.
//!
//! This is the only place unit constants may appear.

/// Boltzmann constant k_B, in Eh/K.
pub const K_B_HARTREE_PER_KELVIN: f64 = 3.1668115634556e-6;

/// One atomic mass unit (dalton), in electron masses m_e.
pub const AMU_IN_ELECTRON_MASSES: f64 = 1822.888486209;

/// One bohr, in ångström.
pub const BOHR_IN_ANGSTROM: f64 = 0.529177210903;

/// One femtosecond, in atomic units of time (ħ/Eh).
pub const FS_IN_ATOMIC_TIME: f64 = 41.341373336561;

/// One hartree, in cm⁻¹.
pub const HARTREE_IN_WAVENUMBERS: f64 = 219474.6313632;

/// One hartree, in kcal/mol.
pub const HARTREE_IN_KCAL_PER_MOL: f64 = 627.5094740631;
