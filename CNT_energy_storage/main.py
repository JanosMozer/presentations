import numpy as np
from ase.build import nanotube
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm

n = 6 # Chirality index n
m = 6  # Chirality index m
length = 6  # Number of unit cells along the tube axis

# --- Mechanical Deformation ---
twist_angle = 0.05  # Radians of twist per Ångström along the z-axis
max_twist = 0.4  # Maximum twist angle for simulation
num_frames = 120   # Number of frames for animation
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
bond_break_distance = 2.5  # Å, distance at which bonds break
bond_reform_distance = 1.8  # Å, distance at which new bonds can form
bond_break_region = 0.1  # Å, region around bond_break_distance for probabilistic breaking


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
    Twist CNT around its central axis (z-direction).
    twist_angle : radians of twist per Å along z
    """
    positions = cnt.get_positions()
    
    # Find the center of the tube in x-y plane
    x_center = np.mean(positions[:, 0])
    y_center = np.mean(positions[:, 1])
    z_min = np.min(positions[:, 2])
    
    new_positions = positions.copy()

    for i, (x, y, z) in enumerate(positions):
        # Translate to center the tube at origin in x-y plane
        x_rel = x - x_center
        y_rel = y - y_center
        
        # Calculate twist angle based on z position
        dz = z - z_min
        theta = twist_angle * dz  # linear twist along z
        
        # Apply rotation around z-axis
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = cos_t * x_rel - sin_t * y_rel + x_center
        y_new = sin_t * x_rel + cos_t * y_rel + y_center
        
        new_positions[i] = [x_new, y_new, z]

    cnt.set_positions(new_positions)
    return cnt

# --- Compute total energy from Morse potential ---
def compute_energy(cnt: Atoms, cutoff, previous_bonds=None):
    """
    Compute energy with bond breaking/reforming logic.
    previous_bonds: list of previous bonds to check for breaking/reforming
    """
    positions = cnt.get_positions()
    bonds = []
    broken_bonds = 0
    reformed_bonds = 0
    
    # If we have previous bonds, check which ones survive
    active_bonds = set()
    if previous_bonds is not None:
        for bond in previous_bonds:
            i, j = bond['indices']
            r = np.linalg.norm(positions[i] - positions[j])
            
            # Probabilistic bond breaking, only for stretched bonds
            bond_survives = True
            if r > (bond_break_distance - bond_break_region):
                if r >= bond_break_distance:
                    # Definitely breaks if beyond break distance
                    bond_survives = False
                else:
                    # Probabilistic breaking in the transition region
                    # Probability increases as we approach bond_break_distance
                    distance_from_start = r - (bond_break_distance - bond_break_region)
                    break_probability = distance_from_start / bond_break_region
                    # Use bond-specific random seed for consistent but varied breaking
                    np.random.seed(hash((i, j, int(r * 1000))) % 2**32)
                    if np.random.random() < break_probability:
                        bond_survives = False
                    # Reset random seed
                    np.random.seed()
            
            if bond_survives:
                bond_energy = morse_potential(r)
                bonds.append({'indices': (i, j), 'energy': bond_energy, 'length': r})
                active_bonds.add((min(i,j), max(i,j)))
            else:
                broken_bonds += 1
    
    # Track bond counts for each atom
    atom_bond_counts = np.zeros(len(cnt), dtype=int)
    for bond in bonds:
        atom_bond_counts[bond['indices'][0]] += 1
        atom_bond_counts[bond['indices'][1]] += 1

    # Look for new bonds to form, respecting the 3-bond limit per atom
    potential_new_bonds = []
    for i in range(len(cnt)):
        # Only consider atoms that have fewer than 3 bonds
        if atom_bond_counts[i] < 3:
            neighbors = []
            for j in range(len(cnt)):
                if i == j: continue
                # Check if the other atom also has capacity for a new bond
                if atom_bond_counts[j] < 3:
                    bond_key = tuple(sorted((i, j)))
                    if bond_key not in active_bonds:
                        r = np.linalg.norm(positions[i] - positions[j])
                        if r < bond_reform_distance:
                            neighbors.append({'distance': r, 'index': j})
            
            # Find the closest valid neighbor to form a new bond with
            if neighbors:
                closest_neighbor = min(neighbors, key=lambda x: x['distance'])
                potential_new_bonds.append(tuple(sorted((i, closest_neighbor['index']))))

    # Add the new bonds, ensuring no duplicates and re-checking bond limits
    for i, j in set(potential_new_bonds):
        if atom_bond_counts[i] < 3 and atom_bond_counts[j] < 3:
            r = np.linalg.norm(positions[i] - positions[j])
            bond_energy = morse_potential(r)
            bonds.append({'indices': (i, j), 'energy': bond_energy, 'length': r})
            active_bonds.add((i, j))
            atom_bond_counts[i] += 1
            atom_bond_counts[j] += 1
            if previous_bonds is not None:
                reformed_bonds += 1
    
    # Final bond list for initial state
    if previous_bonds is None:
        bonds.clear()
        for i in range(len(cnt)):
            for j in range(i + 1, len(cnt)):
                r = np.linalg.norm(positions[i] - positions[j])
                if r < cutoff:
                    bonds.append({'indices': (i, j), 'energy': morse_potential(r), 'length': r})

    # Only count energy from actual bonds, not display-only ones
    total_energy = sum(b['energy'] for b in bonds if not b.get('display_only', False))
    actual_bonds = len([b for b in bonds if not b.get('display_only', False)])
    bond_stats = {'broken': broken_bonds, 'reformed': reformed_bonds, 'total': actual_bonds}
    
    return total_energy, bonds, bond_stats

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
        
        # Different visualization for display-only bonds
        if bond.get('display_only', False):
            # Show broken/stretched bonds as thin gray lines
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='gray', lw=0.5, alpha=0.3)
        else:
            # Normal energy-colored bonds
            color = coolwarm(norm(energy))
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color=color, lw=1)

    # Auto-adjust the view to show the full tube
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    
    # Set equal aspect ratio and proper limits
    max_range = max(x_range, y_range, z_range) * 0.6
    x_center = (np.max(x) + np.min(x)) / 2
    y_center = (np.max(y) + np.min(y)) / 2
    z_center = (np.max(z) + np.min(z)) / 2
    
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)
    ax.set_zlim(z_center - max_range, z_center + max_range)

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
    bond_counts = []
    broken_bond_counts = []
    reformed_bond_counts = []
    max_bond_energy = 0
    previous_bonds = None

    for i, angle in enumerate(twist_angles):
        cnt_twisted = base_cnt.copy()
        cnt_twisted = twist_cnt(cnt_twisted, twist_angle=angle)
        
        energy, bonds, bond_stats = compute_energy(cnt_twisted, cutoff=cutoff, previous_bonds=previous_bonds)
        
        total_energies.append(energy)
        bond_counts.append(bond_stats['total'])
        broken_bond_counts.append(bond_stats['broken'])
        reformed_bond_counts.append(bond_stats['reformed'])
        
        animation_frames.append({'atoms': cnt_twisted, 'bonds': bonds, 'stats': bond_stats})
        
        if bonds:
            max_bond_energy = max(max_bond_energy, max(b['energy'] for b in bonds))
        
        # Store bonds for next iteration
        previous_bonds = bonds
        
        # Print progress for large simulations
        if i % 10 == 0:
            print(f"Frame {i}/{len(twist_angles)}: E={energy:.2f} eV, Bonds={bond_stats['total']}, Broken={bond_stats['broken']}, Reformed={bond_stats['reformed']}")

    # --- Enhanced Plot with Bond Breaking Effects ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Energy plot
    ax1.plot(twist_angles, total_energies, 'b-', linewidth=2, marker='o', markersize=4, label='Total Energy')
    ax1.set_xlabel("Twist Angle (rad/Å)", fontsize=12)
    ax1.set_ylabel("Stored Torsional Energy (eV)", fontsize=12)
    ax1.set_title(f"Energy vs. Twist Angle for CNT ({n},{m}) - Length: {length} units", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    
    # Bond statistics plot
    ax2.plot(twist_angles, bond_counts, 'g-', linewidth=2, marker='s', markersize=3, label='Total Bonds')
    ax2.plot(twist_angles, np.cumsum(broken_bond_counts), 'r-', linewidth=2, marker='^', markersize=3, label='Cumulative Broken')
    ax2.plot(twist_angles, np.cumsum(reformed_bond_counts), 'orange', linewidth=2, marker='v', markersize=3, label='Cumulative Reformed')
    ax2.set_xlabel("Twist Angle (rad/Å)", fontsize=12)
    ax2.set_ylabel("Bond Count", fontsize=12)
    ax2.set_title("Bond Breaking/Reforming Statistics", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("CNT_energy_storage/output/energy_vs_twist.png", dpi=300, bbox_inches='tight')
    # plt.show() # Disabled to not show the plot, only save it.

    # --- Create Animation ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Enable interactive navigation (zoom, pan, rotate)
    ax.mouse_init()

    def update(frame):
        data = animation_frames[frame]
        cnt = data['atoms']
        bonds = data['bonds']
        stats = data['stats']
        angle = twist_angles[frame]
        energy = total_energies[frame]
        title = f"Twisted CNT (Angle={angle:.3f} rad/Å, E={energy:.2f} eV)\nBonds: {stats['total']}, Broken: {stats['broken']}, Reformed: {stats['reformed']}"
        plot_cnt(ax, cnt, bonds, title=title, max_energy=max_bond_energy)

    ani = FuncAnimation(fig, update, frames=len(twist_angles), interval=100, repeat=True)
    
    # Show the interactive animation
    plt.show()
