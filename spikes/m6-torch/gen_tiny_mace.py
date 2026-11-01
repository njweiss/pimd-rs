"""Randomly initialised MACE models for the M6a spike (MILESTONE_6 "Models"; decision 5).

    pixi run -e ml python spikes/m6-torch/gen_tiny_mace.py --kind tiny --out spikes/m6-torch/out/tiny.model

Kinds (all ScaleShiftMACE with RealAgnosticResidualInteractionBlock, elements H, C, N, O, float64):
  tiny             r_max 5 Å, 8x0e+8x1o, max_ell 2, correlation 2, 2 interactions, 4 Bessel,
                   radial MLP [8, 8]: the fast test model (a few kB of weights).
  off23-small-like the MACE-OFF23 "small" hyperparameters (96x0e, max_ell 3, correlation 3,
                   2 interactions, r_max 4.5 Å, 8 Bessel, polynomial cutoff 5, radial MLP
                   [64, 64, 64], MLP 16x0e), random weights. For timing only: NOT MACE-OFF23.
  off23-medium-like as above with 128x0e+128x1o and r_max 5 Å (MACE-OFF23 "medium" shape).

Weights are random (fixed seed), so energies are not physical; they are deterministic functions of
the seed and the mace-torch / torch versions. E0 values (eV) are round numbers of the right size.
The model file is a full pickled module (torch.save), as MACE's own training writes it; it is never
committed (*.model is in .gitignore).
"""

import argparse
import json

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import numpy as np
import torch
from e3nn import o3
from mace import modules

Z = [1, 6, 7, 8]
E0 = [-13.6, -1029.0, -1485.0, -2042.0]  # eV, H C N O

KINDS = {
    "tiny": dict(r_max=5.0, hidden="8x0e+8x1o", max_ell=2, correlation=2, bessel=4,
                 radial_mlp=[8, 8], mlp="8x0e"),
    "off23-small-like": dict(r_max=4.5, hidden="96x0e", max_ell=3, correlation=3, bessel=8,
                             radial_mlp=[64, 64, 64], mlp="16x0e"),
    "off23-medium-like": dict(r_max=5.0, hidden="128x0e+128x1o", max_ell=3, correlation=3,
                              bessel=8, radial_mlp=[64, 64, 64], mlp="16x0e"),
}


def build(kind, seed):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    k = KINDS[kind]
    inter = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    model = modules.ScaleShiftMACE(
        r_max=k["r_max"],
        num_bessel=k["bessel"],
        num_polynomial_cutoff=5,
        max_ell=k["max_ell"],
        interaction_cls=inter,
        interaction_cls_first=inter,
        num_interactions=2,
        num_elements=len(Z),
        hidden_irreps=o3.Irreps(k["hidden"]),
        MLP_irreps=o3.Irreps(k["mlp"]),
        atomic_energies=np.array(E0),
        avg_num_neighbors=6.0,
        atomic_numbers=Z,
        correlation=k["correlation"],
        gate=torch.nn.functional.silu,
        radial_MLP=k["radial_mlp"],
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    # Random init leaves some readout weights small; scale every parameter up a little so the
    # interaction energy and forces are O(1) eV and eV/Å rather than ~1e-3 (fixed, seed-derived).
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(KINDS), default="tiny")
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = build(a.kind, a.seed)
    torch.save(model, a.out)
    n = sum(p.numel() for p in model.parameters())
    print(json.dumps({"kind": a.kind, "seed": a.seed, "parameters": n, "out": a.out,
                      "torch": torch.__version__, **KINDS[a.kind]}))


if __name__ == "__main__":
    main()
