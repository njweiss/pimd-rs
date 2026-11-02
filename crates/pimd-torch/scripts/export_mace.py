"""Export a MACE model to TorchScript for pimd-torch's `MacePotential` (SPEC P-6, MILESTONE_6).

    pixi run -e ml python crates/pimd-torch/scripts/export_mace.py MODEL --out OUT.pt

From the M6a spike (spikes/m6-torch/export_mace.py), unchanged in substance. The adapter expects
the float64 export (the default) and casts it for `Dtype::F32` itself.

MODEL is a MACE model file as MACE writes it (a pickled torch.nn.Module: `mace_run_train` output,
or a downloaded MACE-OFF23 `.model`). The model is wrapped in `PimdMace`, a module with a
positional-tensor interface (no Dict[str, Tensor] marshalling on the Rust side), and compiled with
`e3nn.util.jit.compile`, the route mace's own `run_train` and `create_lammps_model` use.

Interface of the exported module (all lengths Å, energies eV; float tensors in the export dtype):
    forward(positions [n, 3], species [n] int64 (index into atomic_numbers()),
            edge_index [2, E] int64 (row 0 sender i, row 1 receiver j),
            shifts [E, 3] (Å; edge vector = positions[j] - positions[i] + shift),
            ptr [B + 1] int64 (graph b holds atoms ptr[b] .. ptr[b+1]),
            compute_virials: bool,
            unit_shifts [E, 3] (integer image counts, same dtype; only read for virials),
            cell [B, 3, 3] (Å, zeros for gas phase; only read for virials))
      -> (energy [B] eV, forces [n, 3] eV/Å, virial [B, 3, 3] eV; zeros unless compute_virials)
    r_max() -> Tensor [] (Å), atomic_numbers() -> Tensor [S] int64, dtype_code() -> 64 or 32.

Forces are -dE/dpositions by autograd inside the module (MACE `compute_force=True`), so the
caller must keep grad mode enabled (no `no_grad`).
"""

import argparse
import os
from typing import Tuple

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import torch
from e3nn.util import jit


@jit.compile_mode("script")  # e3nn's compile() only scripts modules tagged like this
class PimdMace(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.register_buffer("z", model.atomic_numbers.detach().clone().to(torch.int64))
        self.register_buffer("rmax", model.r_max.detach().clone().to(dtype))
        self.n_species = int(model.atomic_numbers.numel())
        self.code = 64 if dtype == torch.float64 else 32

    def forward(
        self,
        positions: torch.Tensor,
        species: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        ptr: torch.Tensor,
        compute_virials: bool,
        unit_shifts: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_graphs = ptr.numel() - 1
        counts = ptr[1:] - ptr[:-1]
        batch = torch.repeat_interleave(
            torch.arange(n_graphs, device=positions.device), counts
        )
        node_attrs = torch.nn.functional.one_hot(species, self.n_species).to(positions.dtype)
        data = {
            "positions": positions,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell.reshape(-1, 3),
            "batch": batch,
            "ptr": ptr,
        }
        out = self.model(data, training=False, compute_force=True, compute_virials=compute_virials)
        energy = out["energy"]
        forces = out["forces"]
        assert energy is not None
        assert forces is not None
        virials = out["virials"]
        if virials is None:
            virials = torch.zeros((n_graphs, 3, 3), dtype=positions.dtype, device=positions.device)
        return energy, forces, virials

    @torch.jit.export
    def r_max(self) -> torch.Tensor:
        return self.rmax

    @torch.jit.export
    def atomic_numbers(self) -> torch.Tensor:
        return self.z

    @torch.jit.export
    def dtype_code(self) -> int:
        return self.code


def export(model: torch.nn.Module, dtype: torch.dtype = torch.float64):
    """The scripted `PimdMace` wrapper of `model` (cast to `dtype`, parameters frozen)."""
    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return jit.compile(PimdMace(model, dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    model = torch.load(a.model, map_location="cpu", weights_only=False)
    scripted = export(model, dtype)
    scripted.save(a.out)
    print(f"exported {a.model} -> {a.out} ({a.dtype}, {os.path.getsize(a.out)} bytes, "
          f"r_max {float(model.r_max)} Å, Z {model.atomic_numbers.tolist()})")


if __name__ == "__main__":
    main()
