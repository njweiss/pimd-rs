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

Interface v2 (the default; SPEC P-6, MILESTONE_9 "Export interface v2") differs in three points:
forward's energy is MACE's interaction energy (without the reference energies E0), and
    atomic_energies() -> Tensor [S] float64 (E0 per species, eV, in the order of atomic_numbers();
                         float64 in the float32 export as well, read before the cast),
    interface_version() -> 2.
The adapter adds Σ_i E0(Z_i) in f64. Forces and virials are unchanged: MACE takes them from the
interaction energy already. `--interface 1` writes the M6 export (energy = MACE's total energy, no
atomic_energies() or interface_version()). A v2 export is refused for a model without
`atomic_energies_fn`, with more than one head, or whose outputs lack `interaction_energy`
(e.g. plain `MACE`).

Forces are -dE/dpositions by autograd inside the module (MACE `compute_force=True`), so the
caller must keep grad mode enabled (no `no_grad`).
"""

import argparse
import os
import sys
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


@jit.compile_mode("script")
class PimdMaceV2(PimdMace):
    """Interface v2: forward returns the interaction energy; adds atomic_energies() and
    interface_version(). `e0` is the model's E0 table [S] as float64, taken before any cast."""

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype, e0: torch.Tensor):
        super().__init__(model, dtype)
        self.register_buffer("e0", e0)

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
        energy = out["interaction_energy"]
        forces = out["forces"]
        assert energy is not None
        assert forces is not None
        virials = out["virials"]
        if virials is None:
            virials = torch.zeros((n_graphs, 3, 3), dtype=positions.dtype, device=positions.device)
        return energy, forces, virials

    @torch.jit.export
    def atomic_energies(self) -> torch.Tensor:
        return self.e0

    @torch.jit.export
    def interface_version(self) -> int:
        return 2


class Refused(Exception):
    """A model the v2 export cannot wrap (reported without a traceback)."""


def _has_interaction_energy(model: torch.nn.Module, dtype: torch.dtype) -> bool:
    """Whether `model` (already cast and in eval mode) returns `interaction_energy`: one
    evaluation of a single atom of species 0 without edges."""
    n_species = int(model.atomic_numbers.numel())
    data = {
        "positions": torch.zeros((1, 3), dtype=dtype),
        "node_attrs": torch.nn.functional.one_hot(
            torch.zeros(1, dtype=torch.int64), n_species
        ).to(dtype),
        "edge_index": torch.zeros((2, 0), dtype=torch.int64),
        "shifts": torch.zeros((0, 3), dtype=dtype),
        "unit_shifts": torch.zeros((0, 3), dtype=dtype),
        "cell": torch.zeros((3, 3), dtype=dtype),
        "batch": torch.zeros(1, dtype=torch.int64),
        "ptr": torch.tensor([0, 1], dtype=torch.int64),
    }
    out = model(data, training=False, compute_force=False, compute_virials=False)
    return out.get("interaction_energy") is not None


def export(model: torch.nn.Module, dtype: torch.dtype = torch.float64, interface: int = 2):
    """The scripted wrapper of `model` (cast to `dtype`, parameters frozen): `PimdMaceV2` for
    `interface` 2, the M6 `PimdMace` for 1. Raises `Refused` for a v2 export of a model without
    `atomic_energies_fn`, with more than one head, or without an `interaction_energy` output."""
    if interface == 1:
        model = model.to(dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return jit.compile(PimdMace(model, dtype))
    if not hasattr(model, "atomic_energies_fn"):
        raise Refused(
            "the model has no atomic_energies_fn (no E0 table); interface v2 needs it "
            "(use --interface 1 for the old export)"
        )
    table = model.atomic_energies_fn.atomic_energies
    if table.dim() > 1 and table.shape[0] > 1:
        raise Refused(
            f"the model has {table.shape[0]} heads (E0 table {tuple(table.shape)}); "
            "interface v2 supports single-head models only"
        )
    # The E0 table as float64, before the model is cast (exact in the float32 export too).
    e0 = table.detach().clone().to(torch.float64).reshape(-1)
    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if not _has_interaction_energy(model, dtype):
        raise Refused(
            f"the model ({type(model).__name__}) does not return interaction_energy; "
            "interface v2 needs it (use --interface 1 for the old export)"
        )
    return jit.compile(PimdMaceV2(model, dtype, e0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interface", type=int, choices=[1, 2], default=2)
    a = ap.parse_args()
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    model = torch.load(a.model, map_location="cpu", weights_only=False)
    try:
        scripted = export(model, dtype, a.interface)
    except Refused as e:
        print(f"export_mace.py: refused {a.model}: {e}", file=sys.stderr)
        sys.exit(1)
    scripted.save(a.out)
    print(f"exported {a.model} -> {a.out} ({a.dtype}, interface {a.interface}, "
          f"{os.path.getsize(a.out)} bytes, r_max {float(model.r_max)} Å, "
          f"Z {model.atomic_numbers.tolist()})")


if __name__ == "__main__":
    main()
