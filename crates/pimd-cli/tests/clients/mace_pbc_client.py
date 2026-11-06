"""Periodic i-PI protocol MACE force client for the M8a comparison with i-PI (MILESTONE_8 "M8a:
Required tests" 8; data/gen_ipi_water_mace.py).

    pixi run -e ml python crates/pimd-cli/tests/clients/mace_pbc_client.py --address unix:/path \
        MODEL --symbols O H H ... [--device cuda] [--dtype float32]

The locked M6 client (crates/pimd-torch/tests/clients/mace_client.py) is gas phase only, so this is
its periodic sibling. It reuses the protocol code of the locked M5 client
(crates/pimd-socket/tests/clients/ipi_client.py: connect, serve; imported by path, unchanged;
importing it pins OMP/BLAS threads to 1 before torch is imported) and adds a MACE backend: mace's
ASE `MACECalculator` on the model file with periodic boundaries in the orthorhombic cell the
server sends (h, bohr; only its diagonal may be non-zero) and mace's own neighbour list. Each
call builds a fresh Atoms object from the positions and the cell alone and wraps the atoms into
the cell (atom by atom; under periodic boundaries energy and forces do not change), so the client
keeps no state between calls (SPEC D-3 note).

Determinism: torch's deterministic algorithms are switched on (`torch.use_deterministic_algorithms`,
with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch is imported) and the TorchScript profiling
executor is disabled (`torch._C._jit_set_profiling_mode(False)`, as the engine's adapter does,
SPEC P-6: mace's scripted e3nn modules otherwise run a profiled graph in the first two calls and
an optimised one afterwards, which rounds differently). With both, on CUDA float32 the same
positions give bitwise the same results in every call and in every process (checked during test
authoring on the 64-water box: 12 calls, 3 geometries, 2 processes; without them the first two
calls differ from later ones by up to 2e-6 eV/Å). torch runs with 1 intra-op thread.

Units: positions and the cell arrive in bohr, energy and forces leave in Eh and Eh/bohr, converted
with the SPEC U-2 constants (copied here: the client is external to the engine). Virial: zeros
(not used: i-PI runs NVT, and the engine side does not read it for P > 1, K-6).
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IPI = os.path.join(HERE, "../../../pimd-socket/tests/clients/ipi_client.py")
_spec = importlib.util.spec_from_file_location("ipi_client", IPI)
ipi_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ipi_client)  # sets OMP_NUM_THREADS etc. to 1 before torch is imported
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import mace  # noqa: E402,F401  (sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import Atoms  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2
HARTREE_IN_EV = 27.211386245988  # SPEC U-2


class MacePbc:
    def __init__(self, model, symbols, device, dtype):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch._C._jit_set_profiling_mode(False)
        self.calc = MACECalculator(model_paths=model, device=device, default_dtype=dtype)
        self.symbols = symbols

    def compute(self, pos, h):
        if np.any(h - np.diag(np.diag(h)) != 0.0) or np.any(np.diag(h) <= 0.0):
            sys.exit(f"mace_pbc_client: an orthorhombic cell with positive edges is required, got h = {h.tolist()}")
        cell = np.diag(h) * BOHR_IN_ANGSTROM
        atoms = Atoms(self.symbols, positions=pos * BOHR_IN_ANGSTROM, cell=cell, pbc=True)
        atoms.wrap()
        atoms.calc = self.calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        return e / HARTREE_IN_EV, f * (BOHR_IN_ANGSTROM / HARTREE_IN_EV), np.zeros((3, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--patience", type=float, default=120.0)
    ap.add_argument("model")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    model = MacePbc(a.model, a.symbols, a.device, a.dtype)
    sock = ipi_client.connect(a.address, a.patience)
    try:
        return ipi_client.serve(sock, model)
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
