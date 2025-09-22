import numpy as np

def compute_equilibrium_point(uav, T, wd, tol=1e-9, max_iter=100000):
    z_ast, wr_ast = np.zeros(3), uav.br*np.abs(uav.B[:, 3])*T
    x_ast = np.hstack((z_ast, wr_ast))
    err_norm = np.inf
    iter = 0
    while (err_norm > tol) and (iter < max_iter):
        F = function_equilibrium_point(uav, x_ast[0:3], x_ast[3:], T, wd)
        J = jacobian_equilibrium_point(uav, x_ast[3:])
        x_ast = x_ast - np.matmul(np.linalg.inv(J), F)
        err_norm = np.linalg.norm(F)
        iter += 1

    z_ast, wr_ast = x_ast[0:3], x_ast[3:]

    
    if np.any(wr_ast < 0):
        print("WARNING: Negative rotor equilibrium speed detected:", wr_ast)

    if err_norm > tol:
        print('WARNING: Newton method does not converge to Equilibrium point within maximum number of iteration.')
    return z_ast, wr_ast, F

def function_equilibrium_point(uav, z, wr, T, wd):
    F = np.zeros(7)
    F[0] = uav.ct*(-uav.yl1*wr[0]**2 + uav.yl2*wr[1]**2 + uav.yl3*wr[2]**2 - uav.yl4*wr[3]**2) + (uav.Jyy - uav.Jzz)*wd[1]*wd[2]
    F[1] = uav.ct*( uav.xl1*wr[0]**2 - uav.xl2*wr[1]**2 + uav.xl3*wr[2]**2 - uav.xl4*wr[3]**2) + (uav.Jzz - uav.Jxx)*wd[0]*wd[2]
    F[2] = uav.cd*(         wr[0]**2 +         wr[1]**2 -         wr[2]**2 -         wr[3]**2) + (uav.Jxx - uav.Jyy)*wd[0]*wd[1]
    F[3] = -wr[0] + uav.br*(uav.B[0][0]*uav.ki_wx*z[0] + uav.B[0][1]*uav.ki_wy*z[1] + uav.B[0][2]*uav.ki_wz*z[2]) + uav.br*uav.B[0][3]*T
    F[4] = -wr[1] + uav.br*(uav.B[1][0]*uav.ki_wx*z[0] + uav.B[1][1]*uav.ki_wy*z[1] + uav.B[1][2]*uav.ki_wz*z[2]) + uav.br*uav.B[1][3]*T
    F[5] = -wr[2] + uav.br*(uav.B[2][0]*uav.ki_wx*z[0] + uav.B[2][1]*uav.ki_wy*z[1] + uav.B[2][2]*uav.ki_wz*z[2]) + uav.br*uav.B[2][3]*T
    F[6] = -wr[3] + uav.br*(uav.B[3][0]*uav.ki_wx*z[0] + uav.B[3][1]*uav.ki_wy*z[1] + uav.B[3][2]*uav.ki_wz*z[2]) + uav.br*uav.B[3][3]*T
    return F

def jacobian_equilibrium_point(uav, wr):
    J12 = 2*np.array([[-uav.ct*uav.yl1*wr[0],  uav.ct*uav.yl2*wr[1],  uav.ct*uav.yl3*wr[2], -uav.ct*uav.yl4*wr[3]],
                      [ uav.ct*uav.xl1*wr[0], -uav.ct*uav.xl2*wr[1],  uav.ct*uav.xl3*wr[2], -uav.ct*uav.xl4*wr[3]],
                      [ uav.cd*wr[0],          uav.cd*wr[1],         -uav.cd*wr[2],         -uav.cd*wr[3]         ]])
    
    J21 = np.array([[uav.br*uav.B[0][0]*uav.ki_wx, uav.br*uav.B[0][1]*uav.ki_wy, uav.br*uav.B[0][2]*uav.ki_wz],
                    [uav.br*uav.B[1][0]*uav.ki_wx, uav.br*uav.B[1][1]*uav.ki_wy, uav.br*uav.B[1][2]*uav.ki_wz],
                    [uav.br*uav.B[2][0]*uav.ki_wx, uav.br*uav.B[2][1]*uav.ki_wy, uav.br*uav.B[2][2]*uav.ki_wz],
                    [uav.br*uav.B[3][0]*uav.ki_wx, uav.br*uav.B[3][1]*uav.ki_wy, uav.br*uav.B[3][2]*uav.ki_wz]])
    
    J_row_1 = np.hstack((np.zeros((3, 3)), J12))
    J_row_2 = np.hstack((J21, -np.eye(4)))
    J = np.vstack((J_row_1, J_row_2))
    return J