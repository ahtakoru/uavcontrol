import os
import numpy as np

import eq_point
import linearization
import simulation
import tools

class UAV():
    def __init__(self,  Jxx = 1.32e-2, Jyy = 1.25e-2, Jzz = 2.88e-2,
                        Jxy = 0.,      Jxz = 0.,      Jyz = 0.,
                        xl1 = 0.20, xl2 = 0.20, xl3 = 0.20, xl4 = 0.20,
                        yl1 = 0.20, yl2 = 0.20, yl3 = 0.20, yl4 = 0.20,
                        ct = 3.63e-8, cd = 5.11e-10,
                        ar = 100., br = 9000.,
                        kp_q1 = 4.00, kp_q2 = 4.00, kp_q3 = 2.80,
                        kp_wx = 0.15, ki_wx = 0.20, kd_wx = 0.0003,
                        kp_wy = 0.15, ki_wy = 0.20, kd_wy = 0.0003,
                        kp_wz = 0.20, ki_wz = 0.10, kd_wz = 0.,
                        wmaxx = np.deg2rad(220.), wmaxy = np.deg2rad(220.), wmaxz = np.deg2rad(200.),
                        wc = 2.*np.pi*30.0,
                        b11 = np.sqrt(2)/2, b12 = np.sqrt(2)/2, b13 = 1., b14 = 1.,
                        b21 = np.sqrt(2)/2, b22 = np.sqrt(2)/2, b23 = 1., b24 = 1.,
                        b31 = np.sqrt(2)/2, b32 = np.sqrt(2)/2, b33 = 1., b34 = 1.,
                        b41 = np.sqrt(2)/2, b42 = np.sqrt(2)/2, b43 = 1., b44 = 1.,
                        thr_min = 0.15, thr_max = 1.):
        
        # Set parameters 
        self.Jxx, self.Jyy, self.Jzz = Jxx, Jyy, Jzz
        self.Jxy, self.Jxz, self.Jyz = Jxy, Jxz, Jyz
        self.xl1, self.xl2, self.xl3, self.xl4 = xl1, xl2, xl3, xl4
        self.yl1, self.yl2, self.yl3, self.yl4 = yl1, yl2, yl3, yl4
        self.ct, self.cd = ct, cd
        self.ar, self.br = ar, br
        self.kp_q1, self.kp_q2, self.kp_q3 = kp_q1, kp_q2, kp_q3
        self.kp_wx, self.ki_wx, self.kd_wx = kp_wx, ki_wx, kd_wx
        self.kp_wy, self.ki_wy, self.kd_wy = kp_wy, ki_wy, kd_wy
        self.kp_wz, self.ki_wz, self.kd_wz = kp_wz, ki_wz, kd_wz
        self.wmaxx, self.wmaxy, self.wmaxz = wmaxx, wmaxy, wmaxz
        self.wc = wc
        self.b11, self.b12, self.b13, self.b14 = b11, b12, b13, b14
        self.b21, self.b22, self.b23, self.b24 = b21, b22, b23, b24
        self.b31, self.b32, self.b33, self.b34 = b31, b32, b33, b34
        self.b41, self.b42, self.b43, self.b44 = b41, b42, b43, b44
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.set_matrices()

    def set_matrices(self):

        # Define matrices
        self.J = np.diag([self.Jxx, self.Jyy, self.Jzz])
        self.B = np.array([[-self.b11,  self.b12,  self.b13, self.b14],
                           [ self.b21, -self.b22,  self.b23, self.b24],
                           [ self.b31,  self.b32, -self.b33, self.b34],
                           [-self.b41, -self.b42, -self.b43, self.b44]])
        self.G = np.array([[-self.ct*self.yl1,  self.ct*self.yl2,  self.ct*self.yl3, -self.ct*self.yl4],
                           [ self.ct*self.xl1, -self.ct*self.xl2,  self.ct*self.xl3, -self.ct*self.xl4],
                           [ self.cd,           self.cd,          -self.cd,          -self.cd         ],
                           [ self.ct,           self.ct,           self.ct,           self.ct         ]])
        self.Jinv = np.linalg.inv(self.J)
        self.Kpq = np.diag([self.kp_q1, self.kp_q2, self.kp_q3])




if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    uav = UAV()

    T, wd = 0.5, [10, 10, 0]

    uav.kp_q1 = 196

    A = linearization.compute_A_level_two(uav, T)
    vals, vecs = np.linalg.eig(A)

    # Tmin = tools.determine_thrust_min_threshold(uav)
    # print(Tmin)

    # T, wd = 0.5, [10, 10, 0]
    # z_ast, wr_ast, F = eq_point.compute_equilibrium_point(uav, T, wd)

    # A = linearization.compute_A_level_one(uav, T, wd)
    # vals, vecs = np.linalg.eig(A)

    print(vals.real)

    if np.max(vals.real) < 0: print('Stable with ', np.max(vals.real))
    else: print('Unstable')

    # sol = simulation.simulate_level_1(uav, 0.5, [5., 5., 0.])
    # simulation.plot_results_level_1(sol)