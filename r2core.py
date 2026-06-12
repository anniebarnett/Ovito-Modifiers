import numpy as np

def modify(frame, data):
    all_positions = data.particles.positions
    num_particles = len(all_positions)

    # Identify core atoms
    dislocation_core_mask = data.particles['Dislocation'] != -1
    core_positions = data.particles.positions[dislocation_core_mask]

    if len(core_positions) == 0:
        # No dislocations: assign NaN to all atoms
        min_distances = np.full(num_particles, np.nan)
        data.particles_.create_property('r2core', data=min_distances)
        return

    # Compute minimum distance from each atom to dislocation cores
    distances = np.linalg.norm(all_positions[:, np.newaxis] - core_positions, axis=2)
    min_distances = np.min(distances, axis=1)

    # Assign to particle property
    data.particles_.create_property('r2core', data=min_distances)
