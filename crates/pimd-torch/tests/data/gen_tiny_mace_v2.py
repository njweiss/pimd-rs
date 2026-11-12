"""Interface-v2 test exports of the tiny MACE model (MILESTONE_9 "M9a", SPEC P-6 v2 sentence).

    pixi run -e ml python crates/pimd-torch/tests/data/gen_tiny_mace_v2.py --out crates/pimd-torch/tests/data

Reads the committed tiny_mace.model (gen_tiny_mace.py; locked, not rewritten here) and writes to
--out:

  tiny_mace_v2_f64.pt        v2 export (float64) of tiny_mace.model: forward returns the
                             interaction energy, atomic_energies() the E0 table, interface 2
  tiny_mace_bige0_v1_f64.pt  v1 export (float64) of tiny_mace.model with its E0 table replaced by
  tiny_mace_bige0_v2_f64.pt  BIG_E0, and the v2 export of that same model
  stub_v2_ok.pt              small TorchScript stubs (no MACE inside) with the export interface:
  stub_v2_len.pt             forward returns the interaction energy STUB_E_INT eV for every graph
  stub_v2_nan.pt             and zero forces; they differ only in atomic_energies() /
  stub_v2_inf.pt             interface_version() (see STUBS). Used for the E0-sum and the load
  stub_v2_ver3.pt            error tests.
  stub_v2_noe0.pt
  tiny_mace_v2.json          the tables, the stub definitions, parameter checksums, and the
                             float32 premise check of the big-E0 model (compared by
                             validation/verify_reference_data.py)

The export wrappers are written out here, independently of scripts/export_mace.py (whose v2 is the
M9a implementation): `PimdMaceV1` is the M6 `PimdMace` of export_mace.py unchanged; `PimdMaceV2`
differs only in what the brief's "Export interface v2" lists: forward returns
out["interaction_energy"] in place of out["energy"]; `atomic_energies()` returns the model's
`atomic_energies_fn.atomic_energies` as a float64 buffer [S] in the order of `atomic_numbers()`
(registered after the model is cast, so it is float64 in the export; like every floating buffer
it is cast by libtorch's `module.to(Float)`, which is why the adapter must read it before the
float32 cast); `interface_version()` returns 2. Both are compiled with e3nn's `jit.compile`, as
export_mace.py does.

BIG_E0 (eV, H C N O): magnitudes about 2^20 with mixed signs, so that a water molecule (O + 2 H)
sums to about -2069 eV and ethanol (2 C + 6 H + O) to about -2069 eV as well, while every term and
partial sum is about 1e6 eV. A float32 ulp at 1e6 eV is 0.0625 (0.125 at 2e6), at 2e3 eV it is
2.4e-4 eV, so a float32 E0 sum (MACE's `scatter_sum` in the model dtype, the v1 float32 path)
misses the float64 energy by many ulp of |E|. The fractional parts are chosen so that rounding the
table to float32 alone moves water by -0.08 eV and ethanol by -0.10 eV (the `atomic_energies()`
read after the cast mutant). The generator measures the v1 float32 offset on the reference
geometries of mace_ref.json (water, ethanol, the 4-water box; 8 beads each) and fails unless it is
at least 10 float32 ulp of |E| for every bead (brief test 2 premise).

Deterministic for fixed torch / mace-torch versions (recorded in the JSON); CPU, 1 thread.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Tuple

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import numpy as np
import torch
from e3nn.util import jit

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "tiny_mace.model")
REF = os.path.join(HERE, "mace_ref.json")

Z = [1, 6, 7, 8]
TINY_E0 = [-13.6, -1029.0, -1485.0, -2042.0]  # eV, H C N O (gen_tiny_mace.py)
BIG_E0 = [1000000.03, -2000000.05, -1500000.3, -2002069.23]  # eV, H C N O
BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2 (CODATA 2018)

# Stubs: atomic numbers in a non-sorted order (species index != rank of Z), E0 values that are
# not float32-representable, interaction energy exactly representable in float32.
STUB_Z = [8, 1, 6, 7]
STUB_E0 = [-2042.123456789, -13.587654321, -1029.314159265, -1485.271828183]
STUB_E_INT = 0.375
STUB_R_MAX = 5.0
STUBS = {
    "stub_v2_ok": {"version": 2, "e0": STUB_E0},
    "stub_v2_len": {"version": 2, "e0": STUB_E0[:3]},
    "stub_v2_nan": {"version": 2, "e0": [STUB_E0[0], float("nan"), STUB_E0[2], STUB_E0[3]]},
    "stub_v2_inf": {"version": 2, "e0": [STUB_E0[0], STUB_E0[1], STUB_E0[2], float("-inf")]},
    "stub_v2_ver3": {"version": 3, "e0": STUB_E0},
    "stub_v2_noe0": {"version": 2, "e0": None},
}


def _graph(positions, species, n_species, ptr):
    n_graphs = ptr.numel() - 1
    counts = ptr[1:] - ptr[:-1]
    batch = torch.repeat_interleave(torch.arange(n_graphs, device=positions.device), counts)
    node_attrs = torch.nn.functional.one_hot(species, n_species).to(positions.dtype)
    return n_graphs, batch, node_attrs


@jit.compile_mode("script")
class PimdMaceV1(torch.nn.Module):
    """export_mace.py's M6 `PimdMace` (interface v1), unchanged."""

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
class PimdMaceV2(torch.nn.Module):
    """Interface v2 (MILESTONE_9 "Export interface v2")."""

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.register_buffer("z", model.atomic_numbers.detach().clone().to(torch.int64))
        self.register_buffer("rmax", model.r_max.detach().clone().to(dtype))
        e0 = model.atomic_energies_fn.atomic_energies.detach().clone().to(torch.float64)
        self.register_buffer("e0", e0.reshape(-1))
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
        energy = out["interaction_energy"]
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

    @torch.jit.export
    def atomic_energies(self) -> torch.Tensor:
        return self.e0

    @torch.jit.export
    def interface_version(self) -> int:
        return 2


class _StubBase(torch.nn.Module):
    def __init__(self, version: int):
        super().__init__()
        self.register_buffer("z", torch.tensor(STUB_Z, dtype=torch.int64))
        self.register_buffer("rmax", torch.tensor(STUB_R_MAX, dtype=torch.float64))
        self.e_int = STUB_E_INT
        self.version = version

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
        energy = torch.full(
            (n_graphs,), self.e_int, dtype=positions.dtype, device=positions.device
        )
        forces = torch.zeros_like(positions)
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
        return 64

    @torch.jit.export
    def interface_version(self) -> int:
        return self.version


class Stub(_StubBase):
    """Export-interface stub with an `atomic_energies()` table (float64 buffer)."""

    def __init__(self, version: int, e0):
        super().__init__(version)
        self.register_buffer("e0", torch.tensor(e0, dtype=torch.float64))

    @torch.jit.export
    def atomic_energies(self) -> torch.Tensor:
        return self.e0


class StubNoE0(_StubBase):
    """Export-interface stub that claims interface 2 but has no `atomic_energies()`."""


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


def load_model():
    model = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = model.to(torch.float64).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def f32_ulp(x):
    """Spacing of float32 numbers at |x| (2^(floor(log2|x|) - 23))."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 23)


def premise(v1_export, ref):
    """v1 float64 and float32 energies of the big-E0 model on the mace_ref.json geometries (MACE's
    own neighbour list via mace.data.neighborhood, edges as the adapter builds them: senders,
    receivers, shifts in Å); returns per system the minimum over beads of
    |E_f32 - E_f64| / ulp_f32(|E_f64|). Also checks (not stored) that E_f64 is within 1e-8 eV of
    the mace_ref.json energy plus sum_i (BIG_E0 - TINY_E0)(Z_i)."""
    from mace.data.neighborhood import get_neighborhood

    out = {}
    m64 = torch.jit.load(v1_export)
    m32 = torch.jit.load(v1_export)
    m32.to(torch.float32)
    big = dict(zip(Z, BIG_E0))
    tiny = dict(zip(Z, TINY_E0))
    for name in ("water", "ethanol", "waterbox"):
        s = ref["systems"][name]
        p, n = s["n_beads"], s["n_atoms"]
        zs = s["atomic_numbers"]
        species = torch.tensor([Z.index(z) for z in zs], dtype=torch.int64)
        q = np.array(s["positions_bohr"]).reshape(p, n, 3) * BOHR_IN_ANGSTROM
        cell = s["cell_bohr"]
        shift_e0 = 0.0
        for z in zs:
            shift_e0 += big[z] - tiny[z]
        min_ulp, max_dev = math.inf, 0.0
        for j in range(p):
            if cell is None:
                ei, shifts, unit, _ = get_neighborhood(q[j], 5.0)
            else:
                c = np.diag([x * BOHR_IN_ANGSTROM for x in cell])
                ei, shifts, unit, _ = get_neighborhood(q[j], 5.0, pbc=(True,) * 3, cell=c)
            es = {}
            for key, m, dt in (("f64", m64, torch.float64), ("f32", m32, torch.float32)):
                e, _, _ = m(
                    torch.tensor(q[j], dtype=dt),
                    species,
                    torch.tensor(ei, dtype=torch.int64),
                    torch.tensor(shifts, dtype=dt),
                    torch.tensor([0, n], dtype=torch.int64),
                    False,
                    torch.tensor(unit, dtype=dt),
                    torch.zeros((1, 3, 3), dtype=dt),
                )
                es[key] = float(e.detach().to(torch.float64).reshape(-1)[0])
            min_ulp = min(min_ulp, abs(es["f32"] - es["f64"]) / f32_ulp(es["f64"]))
            max_dev = max(max_dev, abs(es["f64"] - (s["energies_ev"][j] + shift_e0)))
        out[name] = {"min_v1_f32_offset_ulp": min_ulp}
        print(f"{name}: v1 float32 offset >= {min_ulp:.1f} ulp; float64 vs mace_ref + dE0 "
              f"{max_dev:.2e} eV")
        assert min_ulp >= 10.0, (name, min_ulp)
        # float64 rounding of partial sums of size up to ~6e6 eV (ulp 9.3e-10 eV): a few ulp.
        assert max_dev <= 1e-8, (name, max_dev)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    torch._C._jit_set_profiling_mode(False)
    paths = {}

    model = load_model()
    sums = checksums(model)
    assert model.atomic_energies_fn.atomic_energies.reshape(-1).tolist() == TINY_E0
    paths["tiny_mace_v2_f64.pt"] = os.path.join(a.out, "tiny_mace_v2_f64.pt")
    jit.compile(PimdMaceV2(model, torch.float64)).save(paths["tiny_mace_v2_f64.pt"])

    big = load_model()
    with torch.no_grad():
        big.atomic_energies_fn.atomic_energies.copy_(
            torch.tensor(BIG_E0, dtype=torch.float64).reshape(
                big.atomic_energies_fn.atomic_energies.shape
            )
        )
    for v, cls in (("v1", PimdMaceV1), ("v2", PimdMaceV2)):
        name = f"tiny_mace_bige0_{v}_f64.pt"
        paths[name] = os.path.join(a.out, name)
        jit.compile(cls(big, torch.float64)).save(paths[name])

    for name, st in STUBS.items():
        mod = StubNoE0(st["version"]) if st["e0"] is None else Stub(st["version"], st["e0"])
        paths[f"{name}.pt"] = os.path.join(a.out, f"{name}.pt")
        torch.jit.script(mod).save(paths[f"{name}.pt"])

    with open(REF) as f:
        ref = json.load(f)
    checks = premise(paths["tiny_mace_bige0_v1_f64.pt"], ref)

    import e3nn

    def num(x):
        return x if math.isfinite(x) else repr(x)

    out = {
        "versions": {
            "torch": torch.__version__,
            "mace-torch": mace.__version__,
            "e3nn": e3nn.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "sha256_tiny_mace.model": sha256(MODEL),
            **{f"sha256_{k}": sha256(p) for k, p in paths.items()},
        },
        "atomic_numbers": Z,
        "tiny_e0_ev": TINY_E0,
        "big_e0_ev": BIG_E0,
        "tiny_parameters": sums,
        "stub": {
            "atomic_numbers": STUB_Z,
            "e0_ev": STUB_E0,
            "interaction_energy_ev": STUB_E_INT,
            "r_max_angstrom": STUB_R_MAX,
            "variants": {
                k: {"version": v["version"],
                    "e0_ev": None if v["e0"] is None else [num(x) for x in v["e0"]]}
                for k, v in STUBS.items()
            },
        },
        "bige0_premise": checks,
    }
    with open(os.path.join(a.out, "tiny_mace_v2.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote " + ", ".join(f"{k} ({os.path.getsize(p)} bytes)" for k, p in paths.items()))


if __name__ == "__main__":
    main()
