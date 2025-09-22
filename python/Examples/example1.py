from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np

from uav import UAV
import linearization
import simulation
import eq_point

os.system('cls' if os.name == 'nt' else 'clear')
uav = UAV(b11 = -0.5, b21 = -0.5, b34 = -0.3)
T, wd = 0.5, [1.0, 0.0, -0.2]



print(uav.B)

z_ast, wr_ast, F = eq_point.compute_equilibrium_point(uav, T, wd)
print("z*  = ", z_ast)
print("wr* = ", wr_ast)


A = linearization.compute_A_level_one(uav, T, wd)
vals, vecs = np.linalg.eig(A)

if np.max(vals.real) < 0: print('Stable with ', np.max(vals.real))
else: print('Unstable')

t_span = np.arange(0, 10, 0.1)
print(t_span.shape)

sol = simulation.simulate_level_1(uav, T, wd, wr0=np.array([500., 500., 500., 500.]), tspan=np.arange(0, 20, 0.1))

simulation.plot_results_level_1(sol, wd, to_deg=True)