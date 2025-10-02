import numpy as np
from ase.build import nanotube
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm
import os

# --- Simulation Parameters ---
# Geometry
n = 6
m = 6
length = 10

# Morse Potential Parameters for C-C bonds
D_e = 4.74  # Bond dissociation energy (eV)
a = 1.79    # Controls the width of the potential well (1/Å)
r0 = 1.42   # Equilibrium bond distance (Å)

# Animation & Twist Parameters
num_frames = 120
max_twist_rate = 0.1  # Total twist in radians per Ångstrom over the simulation

# Bond Dynamics
bond_break_distance = 1.75  # Distance at which bonds start to break (Å)
bond_reform_distance = 1.6   # Max distance for a new bond to form (Å)
bond_break_region = 0.2     # Probabilistic breaking region width (Å)
cutoff = 2.0                # Initial neighbor search cutoff (Å)

# --- Molecular Dynamics Parameters ---
dt = 0.01                     # Timestep for integration (femtoseconds)
mass = 12.011                 # Mass of a carbon atom (amu)
damping = 0.1                 # Damping factor for relaxation
num_relaxation_steps = 30     # Steps to run MD for each frame to relax the structure

# --- Derived Constants ---
twist_rates = np.linspace(0, max_twist_rate, num_frames)

def morse_potential(r):
    return D_e * (1 - np.exp(-a * (r - r0)))**2

def stored_elastic_energy(r):
    """Calculate stored elastic energy: 0 at equilibrium, positive for any deviation"""
    # Both stretched and compressed bonds store elastic energy
    return D_e * (1 - np.exp(-a * (r - r0)))**2

def morse_force(r):
    """Calculate the magnitude of the force derived from the Morse potential."""
    exponent = np.exp(-a * (r - r0))
    return 2 * a * D_e * (1 - exponent) * exponent

ENERGY_AT_BREAK = morse_potential(bond_break_distance)

def build_cnt(n, m, length):
    return nanotube(n, m, length, bond=r0, vacuum=10.0)

def apply_boundary_twist(positions, twist_rate, z_min, z_max, top_atoms):
    """Geometrically twist only the top layer of atoms as a boundary condition."""
    x_center, y_center = np.mean(positions[:, :2], axis=0)
    
    for i in top_atoms:
        x, y, z = positions[i]
        # Use the total twist angle for the top layer
        theta = twist_rate * (z_max - z_min)
        x_rel, y_rel = x - x_center, y - y_center
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = cos_t * x_rel - sin_t * y_rel + x_center
        y_new = sin_t * x_rel + cos_t * y_rel + y_center
        positions[i] = [x_new, y_new, z]

def compute_forces(num_atoms, positions, bonds):
    """Computes the forces on each atom based on the current bonds."""
    forces = np.zeros((num_atoms, 3))
    for bond in bonds:
        i, j = bond['indices']
        p1, p2 = positions[i], positions[j]
        r_vec = p2 - p1
        r = np.linalg.norm(r_vec)
        
        if r > 1e-6: # Avoid division by zero
            force_magnitude = morse_force(r)
            force_vec = force_magnitude * (r_vec / r)
            forces[i] += force_vec
            forces[j] -= force_vec
    return forces

def compute_energy(cnt: Atoms, cutoff, previous_bonds=None):
    positions = cnt.get_positions()
    num_atoms = len(cnt)
    
    if previous_bonds is None:
        active_bonds = [{'indices': (i, j), 'energy': morse_potential(np.linalg.norm(positions[i] - positions[j]))}
                        for i in range(num_atoms) for j in range(i + 1, num_atoms)
                        if np.linalg.norm(positions[i] - positions[j]) < cutoff]
        return sum(b['energy'] for b in active_bonds), active_bonds, active_bonds, {'broken': 0, 'reformed': 0, 'total': len(active_bonds)}

    surviving_bonds, viz_bonds = [], []
    broken_bonds_count, reformed_bonds_count = 0, 0
    atom_bond_counts = np.zeros(num_atoms, dtype=int)
    
    surviving_indices = set()
    for bond in previous_bonds:
        i, j = bond['indices']
        r = np.linalg.norm(positions[i] - positions[j])
        
        # Bond breaking logic
        is_stretched = r > r0 and r > (bond_break_distance - bond_break_region)
        # Probability of breaking increases linearly within the break region
        prob_break = (r - (bond_break_distance - bond_break_region)) / bond_break_region
        breaks = is_stretched and (r >= bond_break_distance or np.random.random() < prob_break)
        
        if not breaks:
            surviving_bonds.append({'indices': (i, j), 'energy': morse_potential(r)})
            atom_bond_counts[i] += 1; atom_bond_counts[j] += 1
            surviving_indices.add(tuple(sorted((i,j))))
        else:
            broken_bonds_count += 1
            viz_bonds.append({'indices': (i, j), 'status': 'broken'})

    potential_new = sorted([
        {'indices': (i, j), 'distance': np.linalg.norm(positions[i] - positions[j])}
        for i in range(num_atoms) for j in range(i + 1, num_atoms)
        if tuple(sorted((i, j))) not in surviving_indices and 
           np.linalg.norm(positions[i] - positions[j]) < bond_reform_distance
    ], key=lambda x: x['distance'])
    
    reformed_bonds = []
    for bond in potential_new:
        i, j = bond['indices']
        if atom_bond_counts[i] < 3 and atom_bond_counts[j] < 3:
            reformed_bonds.append({'indices': (i, j), 'energy': morse_potential(bond['distance'])})
            atom_bond_counts[i] += 1; atom_bond_counts[j] += 1
            reformed_bonds_count += 1
        else:
            viz_bonds.append({'indices': (i, j), 'status': 'potential'})

    active_bonds = surviving_bonds + reformed_bonds
    total_energy = sum(b['energy'] for b in active_bonds)
    bond_stats = {'broken': broken_bonds_count, 'reformed': reformed_bonds_count, 'total': len(active_bonds)}
    
    return total_energy, active_bonds, active_bonds + viz_bonds, bond_stats

if __name__ == "__main__":
    output_dir = "CNT_energy_storage/output"
    os.makedirs(output_dir, exist_ok=True)

    base_cnt = build_cnt(n=n, m=m, length=length)
    # Initialize positions and velocities ONCE, outside the main loop
    current_positions = base_cnt.get_positions()
    current_velocities = np.zeros_like(current_positions)

    # Identify atom layers for boundary conditions
    z_coords = current_positions[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    bottom_atoms = np.where(z_coords < z_min + 0.1)[0]
    top_atoms = np.where(z_coords > z_max - 0.1)[0]

    animation_frames, total_energies, bond_counts, broken_events, reformed_events = [], [], [], [], []
    
    # Get initial bonds from the starting structure
    _, active_bonds, _, _ = compute_energy(base_cnt, cutoff=cutoff)
    
    for i, rate in enumerate(twist_rates):
        # DO NOT reset positions here. We evolve the state from the previous frame.
        
        # Apply the total twist to the top atoms as the boundary condition for the current state
        apply_boundary_twist(current_positions, rate, z_min, z_max, top_atoms)

        # Molecular Dynamics relaxation loop
        for _ in range(num_relaxation_steps):
            # Compute forces based on current bonds
            forces = compute_forces(len(base_cnt), current_positions, active_bonds)
            
            # Apply damping
            forces -= damping * current_velocities
            
            # Zero out forces for fixed atoms
            forces[bottom_atoms, :] = 0.0
            forces[top_atoms, :] = 0.0
            
            # Update velocities and positions (Velocity Verlet integration)
            current_velocities += 0.5 * forces * dt / mass
            current_positions += current_velocities * dt
            
            # In a full MD, you'd re-calculate forces here for the second half of the velocity update.
            # For this relaxation, a simpler integration is sufficient.
            
            # Update bond list dynamically during relaxation for higher accuracy
            _, active_bonds, _, _ = compute_energy(base_cnt, cutoff=cutoff, previous_bonds=active_bonds)
            # We need to update the positions of the ase object for compute_energy to work
            base_cnt.set_positions(current_positions)


        # After relaxation, update the structure and calculate final properties
        cnt_relaxed = base_cnt.copy()
        cnt_relaxed.set_positions(current_positions)
        
        energy, final_active_bonds, drawable_bonds, stats = compute_energy(cnt_relaxed, cutoff=cutoff, previous_bonds=active_bonds)
        active_bonds = final_active_bonds # Carry over the final bond state to the next twist increment
        
        total_energies.append(energy)
        bond_counts.append(stats['total'])
        broken_events.append(stats['broken'])
        reformed_events.append(stats['reformed'])
        animation_frames.append({'atoms': cnt_relaxed, 'bonds': drawable_bonds, 'stats': stats})
        
        if i % 10 == 0:
            print(f"Frame {i+1}/{num_frames}: Twist Rate={rate:.3f}, E={energy:.2f} eV, Bonds={stats['total']}, Broken={stats['broken']}, Reformed={stats['reformed']}")

    # --- Plotting ---
    fig_plot, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
    ax1.plot(twist_rates, total_energies, 'b-', label='Total Energy')
    ax1.set_ylabel("Stored Torsional Energy (eV)"); ax1.set_title(f"Energy vs. Twist Rate for ({n},{m}) CNT"); ax1.grid(True, alpha=0.3); ax1.legend()
    
    bar_width = (twist_rates[1] - twist_rates[0]) if len(twist_rates) > 1 else 0.1
    ax2.bar(twist_rates - bar_width/2, broken_events, width=bar_width, color='red', alpha=0.7, label='Broken Bonds')
    ax2.bar(twist_rates + bar_width/2, reformed_events, width=bar_width, color='orange', alpha=0.7, label='Reformed Bonds')
    ax2.set_xlabel("Twist Rate (rad/Å)"); ax2.set_ylabel("Bond Events per Frame"); ax2.set_title("Bond Breaking and Reformation Events"); ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "energy_vs_twist.png"), dpi=300)
    plt.close(fig_plot)

    # --- Animation ---
    fig_anim = plt.figure(figsize=(12, 10))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    initial_pos = base_cnt.get_positions()
    atom_plot = ax_anim.scatter(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2], c='k', s=20)
    
    max_bonds_to_plot = 0
    if animation_frames:
        max_bonds_to_plot = len(max(animation_frames, key=lambda x: len(x.get('bonds', []))).get('bonds', []))

    bond_plots = [ax_anim.plot([], [], [], lw=1.5)[0] for _ in range(max_bonds_to_plot)]
    
    # Use a fixed normalization range based on physical energy scale
    # This ensures consistent coloring across all frames
    norm = Normalize(vmin=0, vmax=ENERGY_AT_BREAK if ENERGY_AT_BREAK > 0 else 1.0)

    def update(frame):
        data = animation_frames[frame]
        positions = data['atoms'].get_positions()
        atom_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        bonds_in_frame = data.get('bonds', [])
        for i, line in enumerate(bond_plots):
            if i < len(bonds_in_frame):
                bond = bonds_in_frame[i]
                p1, p2 = positions[bond['indices'][0]], positions[bond['indices'][1]]
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]]); line.set_3d_properties([p1[2], p2[2]])
                
                status = bond.get('status', 'active')
                if status == 'active':
                    # Recalculate energy in real-time for accurate coloring
                    current_length = np.linalg.norm(p2 - p1)
                    energy_value = morse_potential(current_length)
                    # We use morse_potential directly here because stored_elastic_energy might be 0
                    # but we want to show the full potential landscape in color.
                    # Let's stick to the physical stored energy definition
                    energy_value = morse_potential(current_length)
                    
                    color_value = norm(energy_value)
                    line.set_color(coolwarm(color_value)); line.set_linestyle('-'); line.set_linewidth(1.5); line.set_alpha(1.0)
                elif status == 'broken':
                    line.set_color('gray'); line.set_linestyle('-'); line.set_linewidth(0.5); line.set_alpha(0.3)
                elif status == 'potential':
                    line.set_color('purple'); line.set_linestyle(':'); line.set_linewidth(0.7); line.set_alpha(0.4)
                line.set_visible(True)
            else:
                line.set_visible(False)
        
        if frame == 0:
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
            ax_anim.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
        
        stats = data['stats']
        ax_anim.set_title(f"Twist Rate: {twist_rates[frame]:.4f} rad/Å\nBonds: {stats['total']}, Broken: {stats['broken']}, Reformed: {stats['reformed']}")

    ani = FuncAnimation(fig_anim, update, frames=num_frames, interval=50, blit=False)
    plt.show()
