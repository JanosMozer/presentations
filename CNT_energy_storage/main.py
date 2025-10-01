import numpy as np
from ase.build import nanotube
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Morse potential parameters for C–C bond ---
# Typical values (from literature / MM3 force fields)
D_e = 6.0  # eV, bond dissociation energy
a = 2.0    # 1/Å, stiffness parameter
r0 = 1.42  # Å, equilibrium bond length (C–C in graphene)

def morse_potential(r):
    """Return Morse potential energy for distance r (in Å)."""
    return D_e * (1 - np.exp(-a * (r - r0)))**2 - D_e

# --- CNT builder ---
def build_cnt(n=6, m=6, length=4):
    """
    Build a CNT using ASE.
    n, m : chirality indices
    length : number of unit cells along tube axis
    """
    cnt = nanotube(n, m, length, bond=1.42)
    return cnt

# --- Apply torsional twist ---
def twist_cnt(cnt: Atoms, twist_angle=0.1):
    """
    Twist CNT around its z-axis.
    twist_angle : radians of twist per Å along z
    """
    positions = cnt.get_positions()
    z_min = np.min(positions[:, 2])
    z_max = np.max(positions[:, 2])
    new_positions = positions.copy()

    for i, (x, y, z) in enumerate(positions):
        dz = z - z_min
        theta = twist_angle * dz  # linear twist
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y
        new_positions[i] = [x_new, y_new, z]

    cnt.set_positions(new_positions)
    return cnt

# --- Compute total energy from Morse potential ---
def compute_energy(cnt: Atoms, cutoff=2.0):
    positions = cnt.get_positions()
    energy = 0.0
    bonds = []
    for i in range(len(cnt)):
        for j in range(i+1, len(cnt)):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < cutoff:  # nearest neighbors
                energy += morse_potential(r)
                bonds.append((i, j))
    return energy, bonds

# --- Visualization ---
def plot_cnt(cnt: Atoms, bonds, title="CNT Structure"):
    positions = cnt.get_positions()
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='k', s=20)

    for (i, j) in bonds:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='b', lw=1)

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    # Build CNT
    cnt = build_cnt(n=6, m=6, length=4)

    # Twist CNT
    cnt_twisted = twist_cnt(cnt, twist_angle=0.05)

    # Compute energy
    energy, bonds = compute_energy(cnt_twisted)
    print(f"Stored torsional energy = {energy:.4f} eV")

    # Plot
    plot_cnt(cnt_twisted, bonds, title=f"Twisted CNT (Energy={energy:.2f} eV)")
