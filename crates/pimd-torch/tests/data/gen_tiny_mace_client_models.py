"""Pickled tiny MACE model files for the MACE socket client tests (MILESTONE_10 "M10a: Required
tests" 2 and 6; SPEC K-7).

    pixi run -e ml python crates/pimd-torch/tests/data/gen_tiny_mace_client_models.py --out crates/pimd-torch/tests/data

The supported MACE socket client (crates/pimd-torch/scripts/mace_client.py) reads a pickled MACE
model (what MACE writes), not a TorchScript export. Its tests compare it with the in-process
adapter on committed v2 exports, so they need the models behind those exports as pickled files.
This script reads the committed tiny_mace.model (gen_tiny_mace.py; locked, not rewritten here),
casts it to float64 and freezes it exactly as gen_tiny_mace_v2.py's `load_model` does, and writes
to --out:

  tiny_mace_bige0.model    its E0 table replaced by gen_tiny_mace_v2.py's BIG_E0 (copied here:
                           H C N O = 1000000.03, -2000000.05, -1500000.3, -2002069.23 eV): the
                           model of the committed export tiny_mace_bige0_v2_f64.pt (test 2)
  tiny_mace_cueq.model     tiny_mace.model with its `use_reduced_cg` attribute set to False (test 6;
                           the model of tiny_mace_v2_f64.pt). tiny_mace.model was built (mace-torch
                           0.3.17) without cuequivariance installed, so its symmetric contractions
                           use the full Clebsch-Gordan basis (mace.tools.cg.U_matrix_real takes the
                           reduced basis only when cuequivariance is importable), while the
                           attribute records the constructor default True. mace's
                           convert_e3nn_cueq trusts the attribute and fails on that file
                           ("einsum(): subscript a has size 7 ... 9"; QUESTIONS.md Q41). The
                           attribute is not read by the e3nn forward; only the converter reads it.
  tiny_mace_client_models.json  E0 tables, flags, parameter checksums and the check below
                           (compared by validation/verify_reference_data.py)

Check (asserted here, values stored): each file just written, reloaded, gives on the water and
ethanol geometries of mace_ref.json (8 beads each, gas phase, edges from mace's own neighbour list)
the same interaction energies and forces as its committed export, to 1e-10 eV and 1e-10 eV/Å (both
float64, same weights; only the summation order inside the scripted and the eager module may
differ, a few ulp of O(1) eV). So each export is the export of that file, and the in-process
adapter on the export is the expected value for the client on the file.

Deterministic for fixed torch / mace-torch versions (recorded in the JSON); CPU, 1 thread.
"""

import argparse
import hashlib
import json
import os
import sys

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "tiny_mace.model")
REF = os.path.join(HERE, "mace_ref.json")

Z = [1, 6, 7, 8]
BIG_E0 = [1000000.03, -2000000.05, -1500000.3, -2002069.23]  # eV, H C N O (gen_tiny_mace_v2.py)
BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2 (CODATA 2018)


def load_model(path):
    model = torch.load(path, map_location="cpu", weights_only=False)
    model = model.to(torch.float64).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def checksums(model):
    ps = [p.detach().to(torch.float64).reshape(-1) for p in model.parameters()]
    flat = torch.cat(ps)
    return {
        "n_parameters": int(flat.numel()),
        "sum": float(flat.sum()),
        "sum_abs": float(flat.abs().sum()),
        "sum_sq": float((flat * flat).sum()),
    }


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check(model_path, export_path, ref):
    """Interaction energies (eV) of the eager model per system and bead, checked against the
    export."""
    from mace.data.neighborhood import get_neighborhood

    model = load_model(model_path)
    export = torch.jit.load(export_path)
    out = {}
    for name in ("water", "ethanol"):
        s = ref["systems"][name]
        p, n = s["n_beads"], s["n_atoms"]
        species = torch.tensor([Z.index(z) for z in s["atomic_numbers"]], dtype=torch.int64)
        q = np.array(s["positions_bohr"]).reshape(p, n, 3) * BOHR_IN_ANGSTROM
        energies = []
        for j in range(p):
            ei, shifts, unit, _ = get_neighborhood(q[j], 5.0)
            args = (
                torch.tensor(q[j], dtype=torch.float64),
                species,
                torch.tensor(ei, dtype=torch.int64),
                torch.tensor(shifts, dtype=torch.float64),
                torch.tensor([0, n], dtype=torch.int64),
                False,
                torch.tensor(unit, dtype=torch.float64),
                torch.zeros((1, 3, 3), dtype=torch.float64),
            )
            e_x, f_x, _ = export(*args)
            data = {
                "positions": args[0],
                "node_attrs": torch.nn.functional.one_hot(species, len(Z)).to(torch.float64),
                "edge_index": args[2],
                "shifts": args[3],
                "unit_shifts": args[6],
                "cell": args[7].reshape(-1, 3),
                "batch": torch.zeros(n, dtype=torch.int64),
                "ptr": args[4],
            }
            o = model(data, training=False, compute_force=True, compute_virials=False)
            e_m = float(o["interaction_energy"].detach().reshape(-1)[0])
            de = abs(e_m - float(e_x.detach().reshape(-1)[0]))
            df = float((o["forces"].detach() - f_x.detach()).abs().max())
            assert de <= 1e-10 and df <= 1e-10, (name, j, de, df)
            energies.append(e_m)
        out[name] = {"interaction_energies_ev": energies}
        print(f"{name}: eager model = export (8 beads)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    torch._C._jit_set_profiling_mode(False)
    with open(REF) as f:
        ref = json.load(f)
    out = {"atomic_numbers": Z, "models": {}}
    versions = {
        "torch": torch.__version__,
        "mace-torch": mace.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "sha256_tiny_mace.model": sha256(MODEL),
    }
    for name, export_name in (
        ("tiny_mace_bige0.model", "tiny_mace_bige0_v2_f64.pt"),
        ("tiny_mace_cueq.model", "tiny_mace_v2_f64.pt"),
    ):
        path = os.path.join(a.out, name)
        model = load_model(MODEL)
        assert model.use_reduced_cg is True
        if name == "tiny_mace_bige0.model":
            with torch.no_grad():
                e0 = model.atomic_energies_fn.atomic_energies
                e0.copy_(torch.tensor(BIG_E0, dtype=torch.float64).reshape(e0.shape))
        else:
            model.use_reduced_cg = False
        torch.save(model, path)
        back = load_model(path)
        assert [int(z) for z in back.atomic_numbers.tolist()] == Z
        table = back.atomic_energies_fn.atomic_energies.detach().reshape(-1).tolist()
        if name == "tiny_mace_bige0.model":
            assert table == BIG_E0, table
        export_path = os.path.join(HERE, export_name)
        out["models"][name] = {
            "export": export_name,
            "e0_ev": table,
            "use_reduced_cg": bool(back.use_reduced_cg),
            "r_max_angstrom": float(back.r_max),
            "parameters": checksums(back),
            "check_vs_export": check(path, export_path, ref),
        }
        versions[f"sha256_{name}"] = sha256(path)
        versions[f"sha256_{export_name}"] = sha256(export_path)
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")

    import e3nn

    versions["e3nn"] = e3nn.__version__
    with open(os.path.join(a.out, "tiny_mace_client_models.json"), "w") as f:
        json.dump({"versions": versions, **out}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
