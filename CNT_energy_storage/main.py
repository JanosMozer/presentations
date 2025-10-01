import numpy as np
from ase.build import nanotube
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm

# --- Simulation Parameters ---
# --- CNT Geometry ---
n = 6 # Chirality index n
m = 6  # Chirality index m
length = 10  # Number of unit cells along the tube axis

# --- Mechanical Deformation ---
twist_angle = 0.05  # Radians of twist per Ångström along the z-axis
max_twist = 0.15  # Maximum twist angle for simulation
num_frames = 60   # Number of frames for animation
twist_angles = np.linspace(0, max_twist, num_frames) # A range of twist angles for the simulation

# --- Morse Potential Parameters for C-C Bond ---
# These parameters define the Morse potential for the Carbon-Carbon bond interactions.
# The Morse potential is a model for the potential energy of a diatomic molecule.
D_e = 6.0  # eV, bond dissociation energy. This is the depth of the potential well.
a = 2.0    # 1/Å, stiffness parameter. This controls the width of the potential well.
r0 = 1.42  # Å, equilibrium bond length. The distance between atoms at which the potential energy is at a minimum.

# --- Energy Calculation ---
cutoff = 2.0 # Å, cutoff distance for considering atomic interactions. 
             # Bonds are only calculated between atoms within this distance.


def morse_potential(r):
    """Return Morse potential energy for distance r (in Å)."""
    # Shift so that equilibrium bond (r0) has 0 energy
    return D_e * (1 - np.exp(-a * (r - r0)))**2

# --- CNT builder ---
def build_cnt(n, m, length):
    """
    Build a CNT using ASE.
    n, m : chirality indices
    length : number of unit cells along tube axis
    """
    cnt = nanotube(n, m, length, bond=r0, vacuum=10.0)
    return cnt

# --- Apply torsional twist ---
def twist_cnt(cnt: Atoms, twist_angle):
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
def compute_energy(cnt: Atoms, cutoff):
    positions = cnt.get_positions()
    bonds = []
    for i in range(len(cnt)):
        for j in range(i + 1, len(cnt)):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < cutoff:  # nearest neighbors
                bond_energy = morse_potential(r)
                bonds.append({'indices': (i, j), 'energy': bond_energy, 'length': r})
    
    total_energy = sum(b['energy'] for b in bonds)
    return total_energy, bonds

# --- Visualization ---
def plot_cnt(ax, cnt: Atoms, bonds, title="CNT Structure", max_energy=1.0):
    positions = cnt.get_positions()
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    ax.clear()
    ax.scatter(x, y, z, c='k', s=20)

    # Handle normalization for bond energy coloring
    if bonds:
        energies = [b['energy'] for b in bonds]
        min_energy = min(energies)
        max_energy = max(energies)
        
        # Handle case where all energies are the same
        if min_energy == max_energy:
            max_energy = min_energy + 1e-6
    else:
        min_energy, max_energy = 0.0, 1.0
    
    norm = Normalize(vmin=min_energy, vmax=max_energy)

    for bond in bonds:
        i, j = bond['indices']
        energy = bond['energy']
        color = coolwarm(norm(energy))
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color=color, lw=1)

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)

# --- Main execution ---
if __name__ == "__main__":
    # Build base CNT
    base_cnt = build_cnt(n=n, m=m, length=length)

    # --- Data Collection Loop ---
    animation_frames = []
    total_energies = []
    max_bond_energy = 0

    for angle in twist_angles:
        cnt_twisted = base_cnt.copy()
        cnt_twisted = twist_cnt(cnt_twisted, twist_angle=angle)
        
        energy, bonds = compute_energy(cnt_twisted, cutoff=cutoff)
        
        total_energies.append(energy)
        animation_frames.append({'atoms': cnt_twisted, 'bonds': bonds})
        
        if bonds:
            max_bond_energy = max(max_bond_energy, max(b['energy'] for b in bonds))

    # --- Plot Energy vs. Twist Angle ---
    plt.figure(figsize=(10, 6))
    plt.plot(twist_angles, total_energies, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel("Twist Angle (rad/Å)", fontsize=12)
    plt.ylabel("Stored Torsional Energy (eV)", fontsize=12)
    plt.title(f"Energy vs. Twist Angle for CNT ({n},{m}) - Length: {length} units", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add some statistics to the plot
    max_energy_idx = np.argmax(total_energies)
    plt.annotate(f'Max: {total_energies[max_energy_idx]:.2f} eV\nat {twist_angles[max_energy_idx]:.3f} rad/Å', 
                xy=(twist_angles[max_energy_idx], total_energies[max_energy_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.savefig("CNT_energy_storage/output/energy_vs_twist.png", dpi=300, bbox_inches='tight')
    # plt.show() # Disabled to not show the plot, only save it.

    # --- Create Animation ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        data = animation_frames[frame]
        cnt = data['atoms']
        bonds = data['bonds']
        angle = twist_angles[frame]
        energy = total_energies[frame]
        plot_cnt(ax, cnt, bonds, title=f"Twisted CNT (Angle={angle:.3f} rad/Å, E={energy:.2f} eV)", max_energy=max_bond_energy)

    ani = FuncAnimation(fig, update, frames=len(twist_angles), interval=100, repeat=True)
    
    # Show the interactive animation
    plt.show()
