from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np

from uav import UAV
import linearization
import simulation

os.system('cls' if os.name == 'nt' else 'clear')


uav = UAV()

T = 0.5
qd = np.array([0.8, 0.3, 0.3, 0.2])
qd = qd/np.linalg.norm(qd)

A2 = linearization.compute_A_level_two(uav, 0.5)
vals, vecs = np.linalg.eig(A2)

if np.max(vals.real) < 0: print('Stable with ', np.max(vals.real))
else: print('Unstable with ', np.max(vals.real))

# Simulate stable system
sol = simulation.simulate_level_2(uav, T, qd)
simulation.plot_results_level_2(sol, qd)

uav.kp_q1 = 200
uav.set_matrices()
A2 = linearization.compute_A_level_two(uav, 0.5)
vals, vecs = np.linalg.eig(A2)

if np.max(vals.real) < 0: print('Stable with ', np.max(vals.real))
else: print('Unstable with ', np.max(vals.real))

# Simulate unstable system
sol = simulation.simulate_level_2(uav, T, qd, wr0=[5000., 5000., 5000., 5000.])
simulation.plot_results_level_2(sol, qd)
