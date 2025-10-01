import numpy as np
from ase.build import nanotube
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm

n = 6; m = 6; length = 10
D_e = 7.37; a = 2.625; r0 = 1.42
num_frames = 120
max_twist_rate = 0.1  # rad/Å
twist_rates = np.linspace(0, max_twist_rate, num_frames)
bond_break_distance = 1.75  # Å
bond_reform_distance = 1.6   # Å
bond_break_region = 0.2     # Å
cutoff = 2.0                # Å

def morse_potential(r):
    return D_e * (1 - np.exp(-a * (r - r0)))**2

ENERGY_AT_BREAK = morse_potential(bond_break_distance)

def build_cnt(n, m, length):
    return nanotube(n, m, length, bond=r0, vacuum=10.0)

def twist_cnt(cnt: Atoms, twist_rate):
    positions = cnt.get_positions()
    x_center, y_center = np.mean(positions[:, :2], axis=0)
    z_min = np.min(positions[:, 2])
    
    new_positions = positions.copy()
    for i, (x, y, z) in enumerate(positions):
        x_rel, y_rel = x - x_center, y - y_center
        theta = twist_rate * (z - z_min)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = cos_t * x_rel - sin_t * y_rel + x_center
        y_new = sin_t * x_rel + cos_t * y_rel + y_center
        new_positions[i] = [x_new, y_new, z]
    cnt.set_positions(new_positions)
    return cnt

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
        is_stretched = r > r0 and r > (bond_break_distance - bond_break_region)
        breaks = is_stretched and (r >= bond_break_distance or np.random.random() < (r - (bond_break_distance - bond_break_region)) / bond_break_region)
        
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
    base_cnt = build_cnt(n=n, m=m, length=length)
    animation_frames, total_energies, bond_counts, broken_events, reformed_events = [], [], [], [], []
    previous_bonds = None

    for i, rate in enumerate(twist_rates):
        cnt_twisted = base_cnt.copy()
        cnt_twisted = twist_cnt(cnt_twisted, twist_rate=rate)
        energy, active_bonds, drawable_bonds, stats = compute_energy(cnt_twisted, cutoff=cutoff, previous_bonds=previous_bonds)
        
        total_energies.append(energy)
        bond_counts.append(stats['total'])
        broken_events.append(stats['broken'])
        reformed_events.append(stats['reformed'])
        animation_frames.append({'atoms': cnt_twisted, 'bonds': drawable_bonds, 'stats': stats})
        previous_bonds = active_bonds
        
        if i % 10 == 0:
            print(f"Frame {i}/{num_frames}: E={energy:.2f} eV, Bonds={stats['total']}, Broken={stats['broken']}, Reformed={stats['reformed']}")

    fig_plot, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
    ax1.plot(twist_rates, total_energies, 'b-', label='Total Energy')
    ax1.set_ylabel("Stored Torsional Energy (eV)"); ax1.set_title(f"Energy vs. Twist Rate for ({n},{m}) CNT"); ax1.grid(True, alpha=0.3); ax1.legend()
    
    bar_width = (twist_rates[1] - twist_rates[0]) * 0.4
    ax2.bar(twist_rates - bar_width/2, broken_events, width=bar_width, color='red', alpha=0.7, label='Broken Bonds')
    ax2.bar(twist_rates + bar_width/2, reformed_events, width=bar_width, color='orange', alpha=0.7, label='Reformed Bonds')
    ax2.set_xlabel("Twist Rate (rad/Å)"); ax2.set_ylabel("Bond Events per Frame"); ax2.set_title("Bond Breaking and Reformation Events"); ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig("CNT_energy_storage/output/energy_vs_twist.png", dpi=300)
    
    fig_anim = plt.figure(figsize=(12, 10))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    initial_pos = base_cnt.get_positions()
    atom_plot = ax_anim.scatter(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2], c='k', s=20)
    
    max_bonds_to_plot = len(max(animation_frames, key=lambda x: len(x['bonds']))['bonds'])
    bond_plots = [ax_anim.plot([], [], [], lw=1.5)[0] for _ in range(max_bonds_to_plot)]
    norm = Normalize(vmin=0, vmax=ENERGY_AT_BREAK)

    def update(frame):
        data = animation_frames[frame]
        positions = data['atoms'].get_positions()
        atom_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        for i, line in enumerate(bond_plots):
            if i < len(data['bonds']):
                bond = data['bonds'][i]
                p1, p2 = positions[bond['indices'][0]], positions[bond['indices'][1]]
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]]); line.set_3d_properties([p1[2], p2[2]])
                
                status = bond.get('status', 'active')
                if status == 'active':
                    line.set_color(coolwarm(norm(bond['energy']))); line.set_linestyle('-'); line.set_linewidth(1.5); line.set_alpha(1.0)
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

    ani = FuncAnimation(fig_anim, update, frames=num_frames, interval=30, blit=False)
    plt.show()
