import numpy as np
from ase.build import nanotube
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm

# --- Simulation Parameters ---
# --- CNT Geometry ---
n = 6  # Chirality index n
m = 6  # Chirality index m
length = 10  # Number of unit cells along the tube axis

# --- Mechanical Deformation ---
twist_angle = 0  # Radians of twist per Ångström along the z-axis

# --- Morse Potential Parameters for C-C Bond ---
D_e = 6.0  # eV, bond dissociation energy
a = 2.0    # 1/Å, stiffness parameter
r0 = 1.42  # Å, equilibrium bond length

# --- Energy Calculation ---
cutoff = 2.0 # Å, cutoff distance for considering atomic interactions

def morse_potential(r):
    """Return Morse potential energy for distance r (in Å)."""
    # Shift so that equilibrium bond (r0) has 0 energy
    return D_e * (1 - np.exp(-a * (r - r0)))**2

def build_cnt(n, m, length):
    """Build a CNT using ASE."""
    return nanotube(n, m, length, bond=r0)

def twist_cnt(cnt: Atoms, twist_angle):
    """Twist CNT around its z-axis."""
    positions = cnt.get_positions()
    z_min = np.min(positions[:, 2])
    new_positions = positions.copy()

    for i, (x, y, z) in enumerate(positions):
        dz = z - z_min
        theta = twist_angle * dz
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y
        new_positions[i] = [x_new, y_new, z]

    cnt.set_positions(new_positions)
    return cnt

def compute_energy(cnt: Atoms, cutoff):
    """Compute total energy and per-bond energies."""
    positions = cnt.get_positions()
    bonds = []
    for i in range(len(cnt)):
        for j in range(i + 1, len(cnt)):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < cutoff:
                bond_energy = morse_potential(r)
                bonds.append({'indices': (i, j), 'energy': bond_energy})
    
    total_energy = sum(b['energy'] for b in bonds)
    return total_energy, bonds

def plot_cnt(ax, cnt: Atoms, bonds, title="CNT Structure"):
    """Plot the CNT with colored bonds."""
    positions = cnt.get_positions()
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    ax.clear()
    ax.scatter(x, y, z, c='k', s=20)

    # Draw all bonds in blue without energy-based coloring
    for bond in bonds:
        i, j = bond['indices']
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue', lw=1.5)

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)
    ax.set_aspect('equal')


if __name__ == "__main__":
    # Build and twist the CNT
    cnt = build_cnt(n=n, m=m, length=length)
    cnt_twisted = twist_cnt(cnt, twist_angle=twist_angle)

    # Compute energy
    energy, bonds = compute_energy(cnt_twisted, cutoff=cutoff)
    print(f"Stored torsional energy = {energy:.4f} eV")

    # Plot the structure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_cnt(ax, cnt_twisted, bonds, title=f"Twisted CNT (Angle={twist_angle:.3f} rad/Å, E={energy:.2f} eV)")
    
    plt.show()
