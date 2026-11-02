"""i-PI protocol MACE force client for the pimd-torch long validation (MILESTONE_6 "Long").

    pixi run -e ml python crates/pimd-torch/tests/clients/mace_client.py --address unix:/path \
        MODEL --symbols O H H [--device cpu] [--dtype float64]

From the M6a spike (spikes/m6-torch/mace_client.py, without its dump option). Reuses the protocol
code of the locked M5 client (crates/pimd-socket/tests/clients/ipi_client.py: connect, serve;
imported by path, unchanged; importing it pins OMP/BLAS threads to 1 before torch is imported) and
adds a MACE backend: mace's ASE `MACECalculator` on the model file, gas phase only (h must be zero,
K-3), torch 1 thread, stateless (a fresh Atoms object per call). Units: positions arrive in bohr,
energy and forces leave in Eh and Eh/bohr, converted with the SPEC U-2 constants (copied here:
the client is external to the engine). Virial: zeros (unused for P > 1, K-6).
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

import mace  # noqa: E402,F401  (sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import Atoms  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2
HARTREE_IN_EV = 27.211386245988  # SPEC U-2


class Mace:
    def __init__(self, model, symbols, device, dtype):
        torch.set_num_threads(1)
        self.calc = MACECalculator(model_paths=model, device=device, default_dtype=dtype)
        self.symbols = symbols

    def compute(self, pos, h):
        if np.any(h != 0.0):
            sys.exit("mace_client: gas phase only (h must be zero)")
        atoms = Atoms(self.symbols, positions=pos * BOHR_IN_ANGSTROM, pbc=False)
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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    a = ap.parse_args()
    model = Mace(a.model, a.symbols, a.device, a.dtype)
    sock = ipi_client.connect(a.address, a.patience)
    try:
        return ipi_client.serve(sock, model)
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
