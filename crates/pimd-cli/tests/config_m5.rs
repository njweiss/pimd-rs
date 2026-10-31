//! Config additions of M5 (MILESTONE_5 "Config format", "Clarifications: Validation of the
//! config"; SPEC C-1, S-2, U-2, U-3, K-1). Expected values are hand conversions with the CODATA
//! 2018 literals of U-2 (not read from `pimd_core::units`), to 1e-15 relative.
//!
//! Error naming (Q15): out-of-range values are `InvalidValue { what: "section.key" }`; type and
//! syntax errors are `Parse`. For the relational rules the brief does not say which of the two
//! applies (e.g. both or neither of `mass_amu`/`masses_amu`), so `rejected_naming` accepts either
//! an `InvalidValue` naming one of the listed fields or a `Parse` whose message mentions one of
//! their keys (assumption, reported).

mod common;

use std::path::PathBuf;

use common::{BASE, parse, remove, set, water_toml};
use pimd_cli::{CliError, Config, PotentialConfig};
use pimd_socket::Address;

const AMU: f64 = 1822.888486209; // m_e (U-2)
const BOHR_A: f64 = 0.529177210903; // Å per bohr (U-2)

fn close(got: f64, want: f64, what: &str) {
    let r = (got - want).abs() / want.abs().max(1e-300);
    assert!(
        r <= 1e-15 || got == want,
        "{what}: {got} vs hand value {want}"
    );
}

fn err(toml: &str) -> CliError {
    match Config::from_toml_str(toml) {
        Ok(c) => panic!("config accepted: {c:?}\n{toml}"),
        Err(e) => e,
    }
}

/// Rejected with `InvalidValue` naming one of `fields`, or `Parse` mentioning one of their keys.
fn rejected_naming(toml: &str, fields: &[&str]) {
    match err(toml) {
        CliError::InvalidValue { what, .. } => {
            assert!(
                fields.contains(&what.as_str()),
                "named {what}, want one of {fields:?}"
            )
        }
        CliError::Parse(m) => assert!(
            fields
                .iter()
                .any(|f| m.contains(f.rsplit('.').next().unwrap())),
            "Parse message {m:?} names none of {fields:?}"
        ),
        other => panic!("expected InvalidValue/Parse naming {fields:?}, got {other:?}"),
    }
}

/// Rejected with exactly `InvalidValue { what: field }`.
fn invalid(toml: &str, field: &str) {
    match err(toml) {
        CliError::InvalidValue { what, .. } => assert_eq!(what, field),
        other => panic!("{field}: expected InvalidValue, got {other:?}"),
    }
}

const WATER_POS: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.1173],
    [0.0, 0.7572, -0.4692],
    [0.0, -0.7572, -0.4692],
];

fn water() -> String {
    water_toml("unix:/tmp/pimd-water.sock", 4)
}

/// C-1, U-3, S-2, K-1 ((new) config, brief example): gas-phase water: per-atom masses
/// 15.999, 1.008, 1.008 amu → × 1822.888486209 m_e; positions Å → ÷ 0.529177210903 bohr; cell
/// `None`; symbols kept; socket potential with the parsed address, n_clients and timeout_s;
/// trajectory_every.
#[test]
fn water_socket_gas_phase() {
    let c = parse(&water());
    let s = &c.system;
    assert_eq!(s.n_atoms, 3);
    assert_eq!(s.masses.len(), 3);
    for (m, a) in s.masses.iter().zip([15.999, 1.008, 1.008]) {
        close(*m, a * AMU, "mass");
    }
    assert!(s.cell.is_none(), "no box_angstrom → gas phase (C-1 None)");
    let pos = s.positions.as_ref().expect("positions");
    assert_eq!(pos.len(), 3);
    for (x, xa) in pos.iter().zip(WATER_POS) {
        for d in 0..3 {
            close(x[d], xa[d] / BOHR_A, "position");
        }
    }
    assert_eq!(
        s.symbols,
        Some(vec!["O".to_string(), "H".to_string(), "H".to_string()])
    );
    assert_eq!(
        c.potential,
        PotentialConfig::Socket {
            address: Address::Unix(PathBuf::from("/tmp/pimd-water.sock")),
            n_clients: 4,
            timeout_s: 20.0,
        }
    );
    assert_eq!(c.output.trajectory_every, Some(10));
    assert!(c.output.rdf.is_none());
}

/// K-1: a TCP address `host:port`, and `n_clients`, `timeout_s` as given.
#[test]
fn socket_tcp_address() {
    let toml = set(
        &set(&water(), "potential", "address", "\"localhost:31415\""),
        "potential",
        "timeout_s",
        "2.5",
    );
    let c = parse(&set(&toml, "potential", "n_clients", "1"));
    assert_eq!(
        c.potential,
        PotentialConfig::Socket {
            address: Address::Tcp("localhost:31415".to_string()),
            n_clients: 1,
            timeout_s: 2.5,
        }
    );
}

/// S-2, C-1 ((new) config): `mass_amu` gives every atom the same mass; `masses_amu` with equal values
/// parses to the same config; `trajectory_every` and `symbols` absent → `None`; M4 configs keep a
/// cell and get no positions (lattice start).
#[test]
fn mass_amu_for_all_atoms_and_defaults() {
    let c = parse(BASE);
    assert_eq!(c.system.masses, vec![20.1797 * AMU; 32]);
    assert!(c.system.cell.is_some());
    assert!(c.system.positions.is_none());
    assert!(c.system.symbols.is_none());
    assert_eq!(c.output.trajectory_every, None);
    let list = format!("[{}]", vec!["20.1797"; 32].join(", "));
    let c2 = parse(&set(
        &remove(BASE, "mass_amu"),
        "system",
        "masses_amu",
        &list,
    ));
    assert_eq!(c, c2, "masses_amu with equal values == mass_amu");
}

/// C-1, C-4 ((new) config): LJ in the gas phase (positions, no box) parses with cell `None`; the
/// C-4 cutoff check needs a cell and does not apply. Positions together with a box are accepted
/// (periodic start from given positions).
#[test]
fn lj_gas_phase_and_positions_with_box() {
    let pos = "[[0.0, 0.0, 0.0], [3.1, 0.0, 0.0], [0.0, 3.1, 0.0], [0.0, 0.0, 3.1]]";
    let four = set(BASE, "system", "n_atoms", "4");
    let four = remove(&remove(&four, "rdf_bins"), "rdf_rmax_angstrom");
    let gas = set(
        &remove(&four, "box_angstrom"),
        "system",
        "positions_angstrom",
        pos,
    );
    let gas = set(&gas, "potential", "cutoff_sigma", "6.0");
    let c = parse(&gas);
    assert!(c.system.cell.is_none());
    let p = c.system.positions.as_ref().unwrap();
    close(p[1][0], 3.1 / BOHR_A, "x");
    assert!(matches!(c.potential, PotentialConfig::Lj(_)));
    let c = parse(&set(&four, "system", "positions_angstrom", pos));
    assert!(c.system.cell.is_some());
    assert!(c.system.positions.is_some());
}

/// S-2 ((new) config validation): exactly one of `mass_amu` / `masses_amu`.
#[test]
fn exactly_one_mass_key() {
    let both = set(&water(), "system", "mass_amu", "1.0");
    rejected_naming(&both, &["system.mass_amu", "system.masses_amu"]);
    let neither = remove(&water(), "masses_amu");
    rejected_naming(&neither, &["system.mass_amu", "system.masses_amu"]);
}

/// S-2 ((new) config validation): `masses_amu` has length `n_atoms` and positive finite values.
#[test]
fn masses_amu_validated() {
    for v in ["[15.999, 1.008]", "[15.999, 1.008, 1.008, 1.008]", "[]"] {
        rejected_naming(
            &set(&water(), "system", "masses_amu", v),
            &["system.masses_amu"],
        );
    }
    for v in [
        "[15.999, -1.008, 1.008]",
        "[15.999, 0.0, 1.008]",
        "[15.999, 1.008, nan]",
    ] {
        invalid(
            &set(&water(), "system", "masses_amu", v),
            "system.masses_amu",
        );
    }
}

/// C-1 ((new) config validation): `positions_angstrom` and `symbols` have length `n_atoms`.
#[test]
fn positions_and_symbols_lengths() {
    let two = "[[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692]]";
    rejected_naming(
        &set(&water(), "system", "positions_angstrom", two),
        &["system.positions_angstrom"],
    );
    let four = "[[0.0, 0.0, 0.1], [0.0, 0.7, -0.4], [0.0, -0.7, -0.4], [1.0, 1.0, 1.0]]";
    rejected_naming(
        &set(&water(), "system", "positions_angstrom", four),
        &["system.positions_angstrom"],
    );
    rejected_naming(
        &set(&water(), "system", "symbols", "[\"O\", \"H\"]"),
        &["system.symbols"],
    );
    rejected_naming(
        &set(
            &water(),
            "system",
            "symbols",
            "[\"O\", \"H\", \"H\", \"H\"]",
        ),
        &["system.symbols"],
    );
}

/// C-1 ((new) config validation): the gas phase (no `box_angstrom`) requires
/// `positions_angstrom` (there is no lattice without a box).
#[test]
fn gas_phase_requires_positions() {
    rejected_naming(
        &remove(&water(), "positions_angstrom"),
        &["system.positions_angstrom", "system.box_angstrom"],
    );
    rejected_naming(
        &remove(BASE, "box_angstrom"),
        &["system.positions_angstrom", "system.box_angstrom"],
    );
}

/// E-7, C-1 ((new) config validation): RDF settings require a box.
#[test]
fn rdf_requires_box() {
    let t = set(
        &set(&water(), "output", "rdf_bins", "10"),
        "output",
        "rdf_rmax_angstrom",
        "3.0",
    );
    rejected_naming(
        &t,
        &[
            "output.rdf_bins",
            "output.rdf_rmax_angstrom",
            "system.box_angstrom",
        ],
    );
}

/// K-1, K-5 ((new) config validation): `n_clients ≥ 1`, `timeout_s > 0` and finite (InvalidValue
/// naming the field); a negative `n_clients` is a type error (`Parse`, Q15); `address` must parse.
#[test]
fn socket_fields_validated() {
    invalid(
        &set(&water(), "potential", "n_clients", "0"),
        "potential.n_clients",
    );
    assert!(matches!(
        err(&set(&water(), "potential", "n_clients", "-1")),
        CliError::Parse(_)
    ));
    for v in ["0.0", "-1.0", "nan", "inf"] {
        invalid(
            &set(&water(), "potential", "timeout_s", v),
            "potential.timeout_s",
        );
    }
    for v in ["\"\"", "\"localhost\"", "\"unix:\"", "\"localhost:port\""] {
        rejected_naming(
            &set(&water(), "potential", "address", v),
            &["potential.address"],
        );
    }
}

/// K-1 ((new) config): a socket section missing `address`, `n_clients` or `timeout_s`, an unknown key
/// in it, or a wrongly typed value is `Parse`.
#[test]
fn socket_section_parse_errors() {
    for key in ["address", "n_clients", "timeout_s"] {
        assert!(
            matches!(err(&remove(&water(), key)), CliError::Parse(_)),
            "missing {key}"
        );
    }
    for t in [
        set(&water(), "potential", "port", "31415"),
        set(&water(), "potential", "n_clients", "\"four\""),
        set(&water(), "output", "trajectory_every", "\"often\""),
        set(&water(), "system", "symbols", "\"OHH\""),
    ] {
        assert!(matches!(err(&t), CliError::Parse(_)), "{t}");
    }
}

/// Q20 (M5 review): `timeout_s` is at most 1e9 s; above that (where a deadline can overflow the
/// clock and panic) it is `InvalidValue { what: "potential.timeout_s" }`, checked at parse time.
#[test]
fn timeout_s_bounded() {
    for v in ["1e9", "1.0", "1e-3"] {
        let c = parse(&set(&water(), "potential", "timeout_s", v));
        match c.potential {
            PotentialConfig::Socket { timeout_s, .. } => {
                assert_eq!(timeout_s, v.parse::<f64>().unwrap())
            }
            other => panic!("expected a socket potential, got {other:?}"),
        }
    }
    for v in ["1.0000001e9", "1e10", "1e19", "1.7e308"] {
        invalid(
            &set(&water(), "potential", "timeout_s", v),
            "potential.timeout_s",
        );
    }
}

/// Q22 (M5 review): every `symbols` entry is non-empty and has no whitespace (it is one field of an
/// xyz line), else `InvalidValue { what: "system.symbols" }`.
#[test]
fn symbols_validated() {
    for v in [
        r#"["", "H", "H"]"#,
        r#"["O H", "H", "H"]"#,
        r#"["O", "H\n", "H"]"#,
        r#"["O", "H", "\tH"]"#,
        "[\"O\", \"H\", \"H\u{a0}\"]",
    ] {
        invalid(&set(&water(), "system", "symbols", v), "system.symbols");
    }
    let c = parse(&set(&water(), "system", "symbols", r#"["O", "H1", "Hé"]"#));
    assert_eq!(
        c.system.symbols,
        Some(vec!["O".to_string(), "H1".to_string(), "Hé".to_string()])
    );
}

/// Q17 (amended in M5): sizes that cannot be allocated are `InvalidValue` at parse time instead of
/// a capacity panic: N·3·8 bytes over `isize::MAX` names `system.n_atoms`; P²·8 bytes (the P×P
/// normal-mode matrix) names `sampling.n_beads`; otherwise P·N·3·8 bytes over `isize::MAX` names
/// `system.n_atoms`. The checks come before any allocation of that size (with `mass_amu`, the
/// per-atom mass vector would otherwise be allocated first). P = 2^30 − 1 is the largest accepted
/// bead count (parse only; nothing of size P² is allocated at parse time).
#[test]
fn unallocatable_sizes_rejected() {
    let isize_max = i64::MAX as u128; // isize::MAX on 64-bit
    let n_limit = isize_max / 24; // largest N with N·3·8 ≤ isize::MAX
    assert_eq!(n_limit * 24 <= isize_max, (n_limit + 1) * 24 > isize_max);
    for n in [n_limit + 1, 1u128 << 62, i64::MAX as u128] {
        invalid(
            &set(BASE, "system", "n_atoms", &n.to_string()),
            "system.n_atoms",
        );
    }
    let p_max: u128 = (1 << 30) - 1;
    assert!(p_max * p_max * 8 <= isize_max && (p_max + 1) * (p_max + 1) * 8 > isize_max);
    for p in [p_max + 1, 1u128 << 32, i64::MAX as u128] {
        invalid(
            &set(BASE, "sampling", "n_beads", &p.to_string()),
            "sampling.n_beads",
        );
    }
    let c = parse(&set(BASE, "sampling", "n_beads", &p_max.to_string()));
    assert_eq!(c.sampling.n_beads as u128, p_max);
    // Neither alone is too large, the product is: P = 2^25, N = 2^40 → P·N·24 = 3·2^68 bytes.
    let t = set(BASE, "system", "n_atoms", &(1u64 << 40).to_string());
    invalid(
        &set(&t, "sampling", "n_beads", &(1u64 << 25).to_string()),
        "system.n_atoms",
    );
}
