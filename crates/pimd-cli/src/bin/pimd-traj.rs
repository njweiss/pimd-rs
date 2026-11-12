//! `pimd-traj <trajectory.xyz>... --out <file.extxyz> [--beads] [--box-angstrom L] [--every n]
//! [--no-wrap] [--bond-angstrom 1.6]` (MILESTONE_9 M9b): converts `trajectory.xyz` files
//! (MILESTONE_5 "Trajectory output"), given in run order, to extended XYZ for OVITO, VMD and ASE.

use std::path::PathBuf;

use clap::Parser;

/// Trajectory export for viewers: centroids (default) or all beads, molecules wrapped.
#[derive(Parser)]
#[command(name = "pimd-traj", version)]
struct Args {
    /// `trajectory.xyz` files in run order (restart parts first).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output extended-XYZ file.
    #[arg(long)]
    out: PathBuf,
    /// Write all P·N beads (bead-major, with a `bead` column) instead of the N centroids.
    #[arg(long)]
    beads: bool,
    /// Cubic box edge L in Å; enables `Lattice`/`pbc` and molecule wrapping.
    #[arg(long = "box-angstrom")]
    box_angstrom: Option<f64>,
    /// Keep every n-th frame.
    #[arg(long, default_value_t = 1)]
    every: usize,
    /// With a box, write positions unwrapped.
    #[arg(long = "no-wrap")]
    no_wrap: bool,
    /// Bond cutoff in Å for finding molecules (minimum image).
    #[arg(long = "bond-angstrom", default_value_t = 1.6)]
    bond_angstrom: f64,
}

fn main() {
    let Args {
        inputs,
        out,
        beads,
        box_angstrom,
        every,
        no_wrap,
        bond_angstrom,
    } = Args::parse();
    let _ = (
        inputs,
        out,
        beads,
        box_angstrom,
        every,
        no_wrap,
        bond_angstrom,
    );
    unimplemented!("pimd-traj (MILESTONE_9 M9b)")
}
