from deepmd.infer import DeepPot
import numpy as np

# =========================
# LOAD MODEL
# =========================
dp = DeepPot("C:\\Users\\ProBook\\Downloads\\model105v3.pb")

# =========================
# CELL (FIXED)
# =========================
cell = np.array([
    [1.2250000247, -2.1217622801,  0.0000000000],
    [1.2250000230,  2.1217622810,  0.0000000000],
    [0.0000000000,  0.0000000000, 20.0000000000]
], dtype=np.float32).reshape(1, -1)

# =========================
# ATOMIC TYPES
# =========================
atype = np.array([0, 0, 0, 0], dtype=np.int32)

# =========================
# STRUCTURE SET
# shape = (n_struct, n_atoms, 3)
# =========================
coords_list = np.array([
    # estructura AA
    [
    [1.225000024,  0.707254094, 11.750000000],
    [1.225000024, -0.707254093, 11.750000000],
    [1.225000024,  0.707254094,  8.250000000],
    [1.225000024, -0.707254093,  8.250000000]
    ],

    # estructura AB 
    [
    [1.225000024,  0.707254094, 11.673371453],
    [1.225000024, -0.707254093, 11.672732045],
    [0.000000000,  0.000000000,  8.326628547],
    [1.225000024, -0.707254093,  8.327267955]
    ]
], dtype=np.float32)

# =========================
# LOOP: SINGLE POINTS
# =========================
energies = []

labels = ["AA", "AB"]


for i, coord in enumerate(coords_list):
    coord = coord.reshape(1, -1)

    e, f, v = dp.eval(coord, cell, atype)

    energy = float(e[0])
    energies.append(energy)
    print(f"Estructura {labels[i]} | Energía = {energy:.8f} eV")

# después del loop donde ya llenaste energies

E_AA = energies[0]
E_AB = energies[1]

delta = (E_AB - E_AA) / 4.0 * 1000.0

print(f"\nΔE/Atom = {delta:.6f} meV")


