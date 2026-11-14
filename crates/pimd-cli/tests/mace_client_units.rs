//! The unit constants of the supported MACE socket client (MILESTONE_10 "M10a: Required tests" 3;
//! SPEC K-7, U-2, U-3).
//!
//! K-7: the client converts bohr → Å, eV → Eh and eV/Å → Eh/bohr with the U-2 constants, copied
//! into the script (it is external to the engine). The values are parsed from the script text and
//! compared bitwise with `pimd_core::units`. Needs no torch, so it is not behind the `torch`
//! feature (it also runs under `CHECK_TORCH=1`). That the script uses these constants, and not
//! others, is checked numerically by `mace_client.rs` (float64 test: a CODATA 2014 Hartree moves
//! the energy by ≈ 5e-6 Eh).

use std::path::Path;

use pimd_core::units::{BOHR_IN_ANGSTROM, HARTREE_IN_EV};

/// The value of `NAME = <float>` at the start of a line of `text`, required exactly once.
fn constant(text: &str, name: &str) -> f64 {
    let values: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix(name))
        .filter_map(|rest| rest.trim_start().strip_prefix('='))
        .map(|v| v.split('#').next().unwrap().trim())
        .collect();
    assert_eq!(
        values.len(),
        1,
        "`{name} = <float>` must be defined exactly once at the start of a line, found {values:?}"
    );
    values[0]
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{name} = {:?}: {e}", values[0]))
}

/// K-7, U-2: `BOHR_IN_ANGSTROM` and `HARTREE_IN_EV` in crates/pimd-torch/scripts/mace_client.py
/// equal `pimd_core::units` bitwise, and the script does not take units from ase (no `ase.units` import or attribute).
#[test]
fn script_constants_equal_core_units_bitwise() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pimd-torch/scripts/mace_client.py");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    for (name, want) in [
        ("BOHR_IN_ANGSTROM", BOHR_IN_ANGSTROM),
        ("HARTREE_IN_EV", HARTREE_IN_EV),
    ] {
        let got = constant(&text, name);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{name}: script {got:e} vs pimd_core::units {want:e}"
        );
    }
    let ase: Vec<&str> = text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("from ase.units") || t.starts_with("import ase.units")
        })
        .collect();
    assert!(
        ase.is_empty() && !text.contains("ase.units."),
        "the client must not take unit constants from ase.units (U-2): {ase:?}"
    );
}
