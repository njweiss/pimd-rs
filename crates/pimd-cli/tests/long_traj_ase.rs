//! M9b required test 4 (readability): the outputs of test 1 (`traj_common::VARIANTS`), read back by
//! ASE `ase.io.read(path, index=":")` from the pixi `ref` environment (ASE 3.29.0 when written),
//! give the hand-written positions, symbols and steps, the cell diag(L, L, L) with pbc all true
//! when a box is given (zero cell, pbc all false without one), and the `bead` array 0 … P−1 with
//! `--beads`. Long test: needs `pixi run -e ref` (feature `long-validation`).
#![cfg(feature = "long-validation")]

mod traj_common;

use std::path::PathBuf;
use std::process::Command;

use traj_common::*;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Prints, for every frame of every file in argv: `frame <step> <pbc as 0/1 ×3> <cell ×9>`, then
/// per atom `atom <symbol> <x> <y> <z> <bead or -1>`; floats with `repr` (shortest round trip).
const READER: &str = r#"
import sys
import ase.io
for path in sys.argv[1:]:
    print("file", path)
    for a in ase.io.read(path, index=":", format="extxyz"):
        cell = [repr(float(v)) for v in a.cell.array.reshape(-1)]
        pbc = [str(int(b)) for b in a.pbc]
        print("frame", int(a.info["step"]), *pbc, *cell)
        bead = a.arrays.get("bead")
        for k, (s, x) in enumerate(zip(a.get_chemical_symbols(), a.positions)):
            b = -1 if bead is None else int(bead[k])
            print("atom", s, *[repr(float(v)) for v in x], b)
"#;

/// M9b test 4: ASE reads every test 1 output with the same positions, cell and `bead` array.
#[test]
fn ase_reads_test1_outputs() {
    let d = tmp();
    let input = write(d.path(), "trajectory.xyz", INPUT);
    let mut outs = Vec::new();
    for (name, args, want) in VARIANTS {
        let out = d.path().join(format!("{name}.extxyz"));
        assert_eq!(convert(&[&input], &out, args), want, "variant {name}");
        outs.push(out);
    }
    let py = root().join(".pixi/envs/ref/bin/python");
    let mut cmd = if py.exists() {
        Command::new(&py)
    } else {
        let mut c = Command::new("pixi");
        c.args(["run", "-e", "ref", "python"]);
        c
    };
    let o = cmd
        .current_dir(root())
        .arg("-c")
        .arg(READER)
        .args(&outs)
        .output()
        .expect("cannot start python (pixi ref env)");
    assert!(
        o.status.success(),
        "ASE reader failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let text = String::from_utf8(o.stdout).unwrap();
    let blocks: Vec<&str> = text.split("file ").filter(|b| !b.is_empty()).collect();
    assert_eq!(blocks.len(), VARIANTS.len());
    for ((name, args, want), block) in VARIANTS.iter().zip(blocks) {
        let boxed = args.contains(&"--box-angstrom");
        let expected = read_extxyz(want);
        let mut lines = block.lines().skip(1);
        for f in &expected {
            let h: Vec<&str> = lines.next().unwrap().split(' ').collect();
            assert_eq!(h[0], "frame", "{name}");
            assert_eq!(h[1].parse::<u64>().unwrap(), f.step(), "{name}: step");
            let pbc = if boxed { "1" } else { "0" };
            assert_eq!(&h[2..5], &[pbc; 3], "{name}: pbc");
            let cell: Vec<f64> = h[5..14].iter().map(|v| v.parse().unwrap()).collect();
            let l = if boxed { 10.0 } else { 0.0 };
            let want_cell = [l, 0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l];
            assert_eq!(cell, want_cell, "{name}: cell");
            for (sym, x, bead) in &f.rows {
                let a: Vec<&str> = lines.next().unwrap().split(' ').collect();
                assert_eq!(a[0], "atom", "{name}");
                assert_eq!(a[1], sym, "{name}: symbol");
                let got: Vec<f64> = a[2..5].iter().map(|v| v.parse().unwrap()).collect();
                assert_eq!(got, x.to_vec(), "{name}: position");
                let b: i64 = a[5].parse().unwrap();
                assert_eq!(b, bead.map_or(-1, |b| b as i64), "{name}: bead");
            }
        }
        assert!(lines.next().is_none(), "{name}: extra frames read by ASE");
    }
}
