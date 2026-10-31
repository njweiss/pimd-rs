//! Restart and config hash after M5 (MILESTONE_5 "Clarifications: Restart (Q18)", "Config hash",
//! "Required tests: Config" hash coverage; SPEC D-3).
//!
//! - The hash covers the new `[system]` fields (masses, cell presence and lengths, initial
//!   positions) and excludes `symbols` (and, M4, `n_steps` and `[output]`, now including
//!   `trajectory_every`). Checked by restarting a 10-step checkpoint under a changed config:
//!   `CliError::Checkpoint` for covered fields, success for excluded ones. (Connection details of
//!   the socket potential: `socket_run.rs`.)
//! - Q18: the restart restores the saved forces and energies instead of re-evaluating them: a
//!   checkpoint whose saved forces were altered is accepted, and the run continues from the
//!   altered forces (its final state differs from the uninterrupted run's).
//!
//! The checkpoint layout used to locate the forces is M4's (format unchanged, version 2;
//! `run.rs` doc of `encode`): magic (8 bytes), version, hash, P, N, step, recorded, Σ E_prim/N,
//! Σ E_vir/N, RNG length (u64 each, little-endian), RNG bytes, q, p, forces (P·N·3 f64 each),
//! energies (P f64), RDF block.

mod common;

use common::{bytes, parse, pool, remove, set, tmp};
use pimd_cli::{CliError, run, run_from_checkpoint};

/// A gas-phase LJ cluster of four atoms with per-atom masses, given positions and symbols.
fn gas(n_steps: usize) -> String {
    format!(
        r#"
[system]
n_atoms = 4
masses_amu = [20.1797, 20.1797, 39.948, 20.1797]
symbols = ["Ne", "Ne", "Ar", "Ne"]
positions_angstrom = [[0.0, 0.0, 0.0], [3.1, 0.0, 0.0], [0.0, 3.1, 0.0], [0.0, 0.0, 3.1]]

[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 6.0
skin_sigma = 0.25
shift = true

[sampling]
temperature_kelvin = 30.0
n_beads = 4
dt_fs = 1.0
tau0_fs = 100.0
n_equil = 5
n_steps = {n_steps}
stride = 5
seed = 99

[output]
checkpoint_every = 10
"#
    )
}

fn restart(toml: &str, ck: &std::path::Path) -> Result<(), CliError> {
    let out = tmp();
    pool(1).install(|| run_from_checkpoint(&parse(toml), ck, out.path()).map(|_| ()))
}

/// D-3 ((new) config hash): changing a mass, an initial position, adding a box (cell presence; 40 Å keeps
/// cutoff + skin = 6.25σ ≤ L/2, C-4, so nothing else changes) or changing the box lengths of a
/// periodic start from positions is `CliError::Checkpoint`.
#[test]
fn hash_covers_new_system_fields() {
    let d = tmp();
    pool(1).install(|| run(&parse(&gas(10)), d.path()).unwrap());
    let ck = d.path().join("checkpoint.bin");
    let g = gas(40);
    let cases = vec![
        (
            "masses_amu",
            set(
                &g,
                "system",
                "masses_amu",
                "[20.1797, 20.1797, 39.948, 20.18]",
            ),
        ),
        (
            "positions_angstrom",
            set(
                &g,
                "system",
                "positions_angstrom",
                "[[0.0, 0.0, 0.0], [3.1, 0.0, 0.0], [0.0, 3.1, 0.0], [0.0, 0.0, 3.2]]",
            ),
        ),
        ("box presence", set(&g, "system", "box_angstrom", "40.0")),
        (
            "cutoff (unchanged M4 coverage)",
            set(&g, "potential", "cutoff_sigma", "5.0"),
        ),
    ];
    for (what, toml) in cases {
        match restart(&toml, &ck) {
            Err(CliError::Checkpoint(_)) => {}
            other => panic!("{what}: expected Checkpoint error, got {other:?}"),
        }
    }
    // Box lengths, for a periodic start from given positions.
    let periodic = |l: &str, n: usize| {
        set(
            &set(&gas(n), "system", "box_angstrom", l),
            "potential",
            "cutoff_sigma",
            "2.5",
        )
    };
    let d2 = tmp();
    pool(1).install(|| run(&parse(&periodic("40.0", 10)), d2.path()).unwrap());
    let ck2 = d2.path().join("checkpoint.bin");
    restart(&periodic("40.0", 40), &ck2).expect("same periodic config restarts");
    match restart(&periodic("[40.0, 40.0, 41.0]", 40), &ck2) {
        Err(CliError::Checkpoint(_)) => {}
        other => panic!("box lengths: expected Checkpoint error, got {other:?}"),
    }
}

/// D-3 ((new) config hash): `symbols`, `trajectory_every`, `n_steps` and the spelling of equal masses
/// (`mass_amu` vs `masses_amu` with equal values) do not change the hash.
#[test]
fn hash_excludes_symbols_and_output() {
    let d = tmp();
    pool(1).install(|| run(&parse(&gas(10)), d.path()).unwrap());
    let ck = d.path().join("checkpoint.bin");
    let g = gas(40);
    for (what, toml) in [
        ("unchanged", g.clone()),
        (
            "symbols changed",
            set(&g, "system", "symbols", "[\"A\", \"B\", \"C\", \"D\"]"),
        ),
        ("symbols removed", remove(&g, "symbols")),
        (
            "trajectory_every",
            set(&g, "output", "trajectory_every", "5"),
        ),
        ("n_steps", gas(25)),
    ] {
        restart(&toml, &ck).unwrap_or_else(|e| panic!("{what}: restart rejected: {e:?}"));
    }
    let uniform = |n: usize| {
        set(
            &remove(&gas(n), "masses_amu"),
            "system",
            "mass_amu",
            "20.1797",
        )
    };
    let d2 = tmp();
    pool(1).install(|| run(&parse(&uniform(10)), d2.path()).unwrap());
    let listed = set(
        &gas(40),
        "system",
        "masses_amu",
        "[20.1797, 20.1797, 20.1797, 20.1797]",
    );
    restart(&listed, &d2.path().join("checkpoint.bin"))
        .expect("masses_amu with equal values restarts a mass_amu checkpoint");
}

/// Byte offset of the saved forces in an M4-format checkpoint for P beads and N atoms.
fn forces_offset(ck: &[u8], p: usize, n: usize) -> usize {
    let rng_len = u64::from_le_bytes(ck[72..80].try_into().unwrap()) as usize;
    80 + rng_len + 2 * p * n * 3 * 8
}

/// D-3, I-3 (Q18): the restart takes the saved forces as they are. Altering one saved force component
/// (+10⁻³ Eh/bohr) in a 20-step checkpoint (M4's periodic BASE system, P = 4, N = 32) is accepted
/// (no re-evaluation check, MILESTONE_5 "Restart"), and the continued run's final checkpoint
/// differs from the uninterrupted run's; with the checkpoint unaltered the two are identical.
#[test]
fn restart_uses_saved_forces() {
    let with = |n: usize| parse(&set(common::BASE, "sampling", "n_steps", &n.to_string()));
    let (full, a, b, c) = (tmp(), tmp(), tmp(), tmp());
    pool(1).install(|| {
        run(&with(40), full.path()).unwrap();
        run(&with(20), a.path()).unwrap();
    });
    let ck = a.path().join("checkpoint.bin");
    let good = bytes(&ck);
    pool(1).install(|| run_from_checkpoint(&with(40), &ck, b.path()).unwrap());
    assert_eq!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&b.path().join("checkpoint.bin"))
    );
    let mut bad = good.clone();
    let off = forces_offset(&good, 4, 32) + 8 * 7;
    let f = f64::from_le_bytes(bad[off..off + 8].try_into().unwrap());
    bad[off..off + 8].copy_from_slice(&(f + 1e-3).to_le_bytes());
    let alt = a.path().join("altered.bin");
    std::fs::write(&alt, &bad).unwrap();
    pool(1)
        .install(|| run_from_checkpoint(&with(40), &alt, c.path()))
        .unwrap_or_else(|e| panic!("Q18: altered forces must be restored, not checked: {e:?}"));
    assert_ne!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&c.path().join("checkpoint.bin")),
        "Q18: the altered saved force had no effect (forces re-evaluated?)"
    );
}
