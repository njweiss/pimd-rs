"""The tiny random MACE test model of pimd-torch (MILESTONE_6 decisions 5 and 9).

    pixi run -e ml python crates/pimd-torch/tests/data/gen_tiny_mace.py --out crates/pimd-torch/tests/data

Writes to --out:
  tiny_mace.model   the pickled MACE module (torch.save), as MACE's own training writes models;
                    the reference data (gen_mace_ref.py) is computed from the committed copy
  tiny_mace_f64.pt  its float64 TorchScript export (crates/pimd-torch/scripts/export_mace.py),
                    made from the file just written (reloaded), so it is the export of that model
  tiny_mace.json    metadata and parameter checksums (compared by validation/verify_reference_data.py)

Model (from the M6a spike's `gen_tiny_mace.py --kind tiny`, unchanged): ScaleShiftMACE with
RealAgnosticResidualInteractionBlock, elements H, C, N, O (Z = 1 6 7 8), r_max 5 Å, hidden irreps
8x0e+8x1o, max_ell 2, correlation 2, two interactions, 4 Bessel functions, polynomial cutoff 5,
radial MLP [8, 8], MLP 8x0e, avg_num_neighbors 6, E0 = -13.6, -1029, -1485, -2042 eV, float64,
torch.manual_seed(20261101), then every parameter += 0.3 * N(0, 1) (same generator) so that
interaction energies and forces are O(0.1-1) eV and eV/Å. Random weights, not physical, not a
third-party model. Deterministic for fixed mace-torch / torch versions (recorded in the JSON).
"""

import argparse
import hashlib
import json
import os
import sys

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import numpy as np
import torch
from e3nn import o3
from mace import modules

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import export_mace  # noqa: E402

SEED = 20261101
Z = [1, 6, 7, 8]
E0 = [-13.6, -1029.0, -1485.0, -2042.0]  # eV, H C N O
R_MAX = 5.0  # Å


def build():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(SEED)
    inter = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    model = modules.ScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=inter,
        interaction_cls_first=inter,
        num_interactions=2,
        num_elements=len(Z),
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.array(E0),
        avg_num_neighbors=6.0,
        atomic_numbers=Z,
        correlation=2,
        gate=torch.nn.functional.silu,
        radial_MLP=[8, 8],
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return model.eval()


def checksums(model):
    """Parameter count and sums over all parameters (float64), in state_dict order."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    model_path = os.path.join(a.out, "tiny_mace.model")
    export_path = os.path.join(a.out, "tiny_mace_f64.pt")
    torch.save(build(), model_path)
    # Export the file just written, not the in-memory object.
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    sums = checksums(model)
    export_mace.export(model, torch.float64).save(export_path)
    import e3nn
    import mace as mace_pkg

    out = {
        "versions": {
            "torch": torch.__version__,
            "mace-torch": mace_pkg.__version__,
            "e3nn": e3nn.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "sha256_tiny_mace.model": sha256(model_path),
            "sha256_tiny_mace_f64.pt": sha256(export_path),
        },
        "seed": SEED,
        "atomic_numbers": Z,
        "e0_ev": E0,
        "r_max_angstrom": R_MAX,
        "parameters": sums,
    }
    with open(os.path.join(a.out, "tiny_mace.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {model_path} ({os.path.getsize(model_path)} bytes), {export_path} "
          f"({os.path.getsize(export_path)} bytes); {sums['n_parameters']} parameters")


if __name__ == "__main__":
    main()
