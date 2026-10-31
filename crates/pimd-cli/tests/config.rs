//! Config parsing tests (MILESTONE_4 "Config format", "Required tests: Config"; SPEC U-2, U-3,
//! U-4, C-4, RP-1, RP-4). Expected values are hand conversions with the CODATA 2018 constants of
//! U-2 (written out as literals here, not taken from `pimd_core::units`), to 1e-15 relative (a
//! conversion by multiplication or by division by the inverse may differ in the last bit).

mod common;

use common::{BASE, cell_of, lj_params, parse, remove, set};
use pimd_cli::{CliError, Config};
use pimd_core::normal_modes::Transform;

/// U-2 literals (CODATA 2018), repeated here so the test does not read the constants it checks.
const K_B: f64 = 3.1668115634556e-6; // Eh/K
const AMU: f64 = 1822.888486209; // m_e
const BOHR_A: f64 = 0.529177210903; // Å per bohr
const FS: f64 = 41.341373336561; // ħ/Eh per fs

fn close(got: f64, want: f64, what: &str) {
    let r = (got - want).abs() / want.abs();
    assert!(r <= 1e-15, "{what}: {got} vs hand value {want} (rel {r:e})");
}

fn err(toml: &str) -> CliError {
    match Config::from_toml_str(toml) {
        Ok(c) => panic!("config accepted: {c:?}"),
        Err(e) => e,
    }
}

/// Asserts `toml` is rejected with `InvalidValue { what, .. }` naming exactly `field`, and returns
/// the value it carries.
fn invalid(toml: &str, field: &str) -> f64 {
    match err(toml) {
        CliError::InvalidValue { what, value } => {
            assert_eq!(what, field, "wrong field named");
            value
        }
        other => panic!("{field}: expected InvalidValue, got {other:?}"),
    }
}

/// U-3, U-4 and "Config format": the full example config parses, and every quantity is converted
/// to atomic units by hand values: ε = 35.6 K → 35.6 k_B Eh; σ = 2.749 Å → 2.749/0.529177210903
/// bohr; cutoff = 2.0 σ, skin = 0.25 σ (bohr); m = 20.1797 amu → 20.1797 × 1822.888486209 m_e;
/// L = 12.5 Å → bohr; β = 1/(30 k_B); dt = 1 fs → 41.341373336561 ħ/Eh; τ_0 = 100 fs; r_max = 6 Å.
#[test]
fn full_config_parses_with_hand_conversions() {
    let c = parse(BASE);
    assert_eq!(c.system.n_atoms, 32);
    assert_eq!(c.system.masses.len(), 32);
    for &m in &c.system.masses {
        close(m, 20.1797 * AMU, "mass");
    }
    for l in cell_of(&c).lengths() {
        close(l, 12.5 / BOHR_A, "box length");
    }
    let sigma = 2.749 / BOHR_A;
    let lj = lj_params(&c);
    close(lj.epsilon, 35.6 * K_B, "epsilon");
    close(lj.sigma, sigma, "sigma");
    close(lj.cutoff, 2.0 * sigma, "cutoff");
    close(lj.skin, 0.25 * sigma, "skin");
    assert!(lj.shift);
    let s = &c.sampling;
    assert_eq!(s.temperature_kelvin, 30.0);
    close(s.beta, 1.0 / (30.0 * K_B), "beta");
    assert_eq!(s.n_beads, 4);
    close(s.dt, 1.0 * FS, "dt");
    close(s.tau0, 100.0 * FS, "tau0");
    assert_eq!((s.n_equil, s.n_steps, s.stride, s.seed), (20, 40, 5, 12345));
    assert_eq!(s.normal_modes, Transform::Matrix, "default must be matrix");
    let rdf = c.output.rdf.as_ref().expect("rdf configured");
    assert_eq!(rdf.n_bins, 20);
    close(rdf.r_max, 6.0 / BOHR_A, "rdf r_max");
    assert_eq!(c.output.checkpoint_every, 10);
}

/// "Config format": the brief's own example (N = 108, 17 Å box, cutoff 2.5σ, skin 0.3σ, P = 16)
/// parses, including `normal_modes = "matrix"` and the large step counts.
#[test]
fn brief_example_parses() {
    let toml = r#"
[system]
n_atoms = 108
mass_amu = 20.1797              # neon
box_angstrom = 17.0             # cubic; or three values
[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.5
skin_sigma = 0.3
shift = true
[sampling]
temperature_kelvin = 30.0
n_beads = 16
dt_fs = 1.0
tau0_fs = 100.0
n_equil = 5000
n_steps = 20000
stride = 10
seed = 12345
normal_modes = "matrix"         # optional; "matrix" (default) or "fft"
[output]
rdf_bins = 100
rdf_rmax_angstrom = 8.0
checkpoint_every = 1000
"#;
    let c = parse(toml);
    assert_eq!(c.system.n_atoms, 108);
    assert_eq!(c.sampling.n_beads, 16);
    assert_eq!(c.sampling.n_steps, 20000);
    close(lj_params(&c).cutoff, 2.5 * 2.749 / BOHR_A, "cutoff");
    assert_eq!(c.output.rdf.as_ref().unwrap().n_bins, 100);
}

/// RP-4 (MILESTONE_4 transform option): `normal_modes` = "matrix" | "fft"; absent → Matrix;
/// anything else is a parse error.
#[test]
fn normal_modes_setting() {
    let fft = parse(&set(BASE, "sampling", "normal_modes", "\"fft\""));
    assert_eq!(fft.sampling.normal_modes, Transform::Fft);
    let mat = parse(&set(BASE, "sampling", "normal_modes", "\"matrix\""));
    assert_eq!(mat.sampling.normal_modes, Transform::Matrix);
    assert_eq!(mat, parse(BASE), "explicit default must equal absent");
    assert!(matches!(
        err(&set(BASE, "sampling", "normal_modes", "\"dft\"")),
        CliError::Parse(_)
    ));
}

/// C-1, "Config format": `box_angstrom` takes one value (cube) or three (orthorhombic).
#[test]
fn box_one_or_three_values() {
    let c = parse(&set(BASE, "system", "box_angstrom", "[12.5, 13.0, 14.0]"));
    let l = cell_of(&c).lengths();
    close(l[0], 12.5 / BOHR_A, "Lx");
    close(l[1], 13.0 / BOHR_A, "Ly");
    close(l[2], 14.0 / BOHR_A, "Lz");
    let cube = parse(&set(BASE, "system", "box_angstrom", "[12.5, 12.5, 12.5]"));
    assert_eq!(cube, parse(BASE));
    assert!(Config::from_toml_str(&set(BASE, "system", "box_angstrom", "[12.5, 13.0]")).is_err());
}

/// "Config format": without `rdf_bins` and `rdf_rmax_angstrom`, `output.rdf` is `None`; giving
/// only one of the two is an error.
#[test]
fn rdf_keys_optional_together() {
    let none = parse(&remove(&remove(BASE, "rdf_bins"), "rdf_rmax_angstrom"));
    assert!(none.output.rdf.is_none());
    assert!(Config::from_toml_str(&remove(BASE, "rdf_bins")).is_err());
    assert!(Config::from_toml_str(&remove(BASE, "rdf_rmax_angstrom")).is_err());
}

/// "Required tests: Config": every required field, when missing, is a `Parse` error. (M5:
/// `mass_amu` and `box_angstrom` are no longer unconditionally required, MILESTONE_5 "Config
/// format"; their missing cases are covered in `config_m5.rs`.)
#[test]
fn missing_fields_are_parse_errors() {
    for key in [
        "n_atoms",
        "kind",
        "epsilon_kelvin",
        "sigma_angstrom",
        "cutoff_sigma",
        "skin_sigma",
        "shift",
        "temperature_kelvin",
        "n_beads",
        "dt_fs",
        "tau0_fs",
        "n_equil",
        "n_steps",
        "stride",
        "seed",
    ] {
        match err(&remove(BASE, key)) {
            CliError::Parse(_) => {}
            other => panic!("missing {key}: expected Parse, got {other:?}"),
        }
    }
}

/// "Config format": unknown keys (in any section), unknown sections, a wrong type, an unknown
/// potential kind and TOML syntax errors are `Parse` errors.
#[test]
fn unknown_keys_and_bad_syntax_are_parse_errors() {
    for toml in [
        set(BASE, "system", "colour", "\"blue\""),
        set(BASE, "potential", "epsilon_kjmol", "1.0"),
        set(BASE, "sampling", "nbeads", "4"),
        set(BASE, "output", "dump_every", "3"),
        format!("{BASE}\n[extra]\nx = 1\n"),
        set(BASE, "sampling", "n_beads", "\"four\""),
        set(BASE, "potential", "kind", "\"morse\""),
        BASE.replace("n_atoms = 32", "n_atoms = = 32"),
    ] {
        match err(&toml) {
            CliError::Parse(_) => {}
            other => panic!("expected Parse, got {other:?} for\n{toml}"),
        }
    }
}

/// U-4, S-2, C-1, I-5, T-2, P-3 ("negative values … rejected"): each negative physical value is an
/// `InvalidValue` naming its field (`section.key`); the value carried is negative.
#[test]
fn negative_values_rejected_naming_the_field() {
    for (section, key, value) in [
        ("system", "mass_amu", "-20.1797"),
        ("system", "box_angstrom", "-12.5"),
        ("potential", "epsilon_kelvin", "-35.6"),
        ("potential", "sigma_angstrom", "-2.749"),
        ("potential", "cutoff_sigma", "-2.0"),
        ("potential", "skin_sigma", "-0.25"),
        ("sampling", "temperature_kelvin", "-30.0"),
        ("sampling", "dt_fs", "-1.0"),
        ("sampling", "tau0_fs", "-100.0"),
        ("output", "rdf_rmax_angstrom", "-6.0"),
    ] {
        let v = invalid(&set(BASE, section, key, value), &format!("{section}.{key}"));
        assert!(v < 0.0, "{section}.{key}: value {v}");
    }
}

/// U-4, I-5: zero or non-finite temperature and time step are `InvalidValue` naming the field.
#[test]
fn zero_and_nonfinite_rejected() {
    for (section, key, value) in [
        ("sampling", "temperature_kelvin", "0.0"),
        ("sampling", "temperature_kelvin", "nan"),
        ("sampling", "temperature_kelvin", "inf"),
        ("sampling", "dt_fs", "0.0"),
        ("system", "mass_amu", "0.0"),
    ] {
        invalid(&set(BASE, section, key, value), &format!("{section}.{key}"));
    }
}

/// RP-1: `n_beads = 0` is `InvalidValue { what: "sampling.n_beads", value: 0 }`.
#[test]
fn zero_beads_rejected() {
    let v = invalid(&set(BASE, "sampling", "n_beads", "0"), "sampling.n_beads");
    assert_eq!(v, 0.0);
}

/// C-4: cutoff + skin > L/2 is rejected at parse time with an `InvalidValue` whose field names the
/// cutoff (`potential.cutoff_sigma` or `potential.skin_sigma`). 2.0σ + 0.3σ = 6.32 Å > 6.25 Å;
/// the boundary case 2.0σ + 0.25σ = 6.185 Å (BASE) is accepted. A non-cubic box uses L_min.
#[test]
fn cutoff_plus_skin_beyond_half_box_rejected() {
    let check = |toml: &str| match err(toml) {
        CliError::InvalidValue { what, .. } => {
            assert!(
                what == "potential.cutoff_sigma" || what == "potential.skin_sigma",
                "C-4 error names {what}"
            );
            what
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    };
    check(&set(BASE, "potential", "skin_sigma", "0.3"));
    check(&set(BASE, "potential", "cutoff_sigma", "2.5"));
    check(&set(BASE, "system", "box_angstrom", "[20.0, 12.0, 20.0]"));
    parse(&set(BASE, "system", "box_angstrom", "[20.0, 12.5, 20.0]"));
}

/// E-7: an RDF range beyond L_min/2 is an error (6.3 Å > 6.25 Å); zero bins is an error.
#[test]
fn rdf_range_and_bins_validated() {
    assert!(Config::from_toml_str(&set(BASE, "output", "rdf_rmax_angstrom", "6.3")).is_err());
    assert!(Config::from_toml_str(&set(BASE, "output", "rdf_bins", "0")).is_err());
}

/// "Required tests: Config" (distinct errors): a missing field, a negative value, n_beads = 0 and
/// cutoff + skin > L/2 give four distinguishable errors (variant and named field).
#[test]
fn errors_are_distinct() {
    let describe = |e: CliError| match e {
        CliError::Parse(_) => "parse".to_string(),
        CliError::InvalidValue { what, .. } => format!("invalid {what}"),
        other => format!("{other:?}"),
    };
    let kinds = [
        describe(err(&remove(BASE, "n_beads"))),
        describe(err(&set(BASE, "sampling", "temperature_kelvin", "-1.0"))),
        describe(err(&set(BASE, "sampling", "n_beads", "0"))),
        describe(err(&set(BASE, "potential", "skin_sigma", "0.3"))),
    ];
    for a in 0..kinds.len() {
        for b in a + 1..kinds.len() {
            assert_ne!(kinds[a], kinds[b], "errors {a} and {b} coincide");
        }
    }
}

/// E-7 (N ≥ 2), "Config format" (N2 after review): with an RDF configured, n_atoms = 1 is
/// rejected at parse time as `InvalidValue` naming `system.n_atoms`; without the RDF keys a
/// single atom is accepted.
#[test]
fn single_atom_with_rdf_rejected() {
    let one = set(BASE, "system", "n_atoms", "1");
    invalid(&one, "system.n_atoms");
    let no_rdf = remove(&remove(&one, "rdf_bins"), "rdf_rmax_angstrom");
    assert!(
        Config::from_toml_str(&no_rdf).is_ok(),
        "one atom without RDF must be accepted"
    );
}
