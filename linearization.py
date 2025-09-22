import numpy as np
import eq_point

def compute_A_level_one(uav, T, wd):
        wdx, wdy, wdz = wd
        _, wr_ast, _ = eq_point.compute_equilibrium_point(uav, T, wd)
        wr1_ast, wr2_ast, wr3_ast, wr4_ast = wr_ast

        A = np.zeros([16, 16])

        A[0, 0]   = (2*uav.Jxy**2*uav.Jyz*wdx - uav.Jxz*uav.Jyy**2*wdy - 2*uav.Jxz**2*uav.Jyz*wdx + 2*uav.Jxy*uav.Jyz**2*wdz - 2*uav.Jxz*uav.Jyz**2*wdy + uav.Jxy*uav.Jzz**2*wdz - uav.Jxx*uav.Jxy*uav.Jyz*wdy + uav.Jxx*uav.Jxz*uav.Jyy*wdy - 2*uav.Jxy*uav.Jxz*uav.Jyy*wdx + uav.Jxx*uav.Jxz*uav.Jyz*wdz - uav.Jxx*uav.Jxy*uav.Jzz*wdz + 2*uav.Jxy*uav.Jxz*uav.Jzz*wdx + uav.Jxy*uav.Jyy*uav.Jyz*wdy - uav.Jxz*uav.Jyy*uav.Jyz*wdz - uav.Jxy*uav.Jyy*uav.Jzz*wdz + uav.Jxy*uav.Jyz*uav.Jzz*wdy + uav.Jxz*uav.Jyy*uav.Jzz*wdy - uav.Jxz*uav.Jyz*uav.Jzz*wdz)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[0, 1]   = (uav.Jxz**2*uav.Jyy*wdz - uav.Jxz*uav.Jyy**2*wdx - 2*uav.Jxz*uav.Jyz**2*wdx - 2*uav.Jxy**2*uav.Jyz*wdy - 2*uav.Jyz**3*wdy + uav.Jyy*uav.Jyz**2*wdz - uav.Jxy**2*uav.Jzz*wdz + uav.Jyy*uav.Jzz**2*wdz - uav.Jyy**2*uav.Jzz*wdz - uav.Jyz**2*uav.Jzz*wdz - uav.Jxx*uav.Jxy*uav.Jyz*wdx + uav.Jxx*uav.Jxz*uav.Jyy*wdx + 2*uav.Jxy*uav.Jxz*uav.Jyy*wdy + uav.Jxy*uav.Jyy*uav.Jyz*wdx + uav.Jxy*uav.Jyz*uav.Jzz*wdx + uav.Jxz*uav.Jyy*uav.Jzz*wdx + 2*uav.Jyy*uav.Jyz*uav.Jzz*wdy)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[0, 2]   = -(uav.Jxy**2*uav.Jzz*wdy - 2*uav.Jxy*uav.Jyz**2*wdx - uav.Jxz**2*uav.Jyy*wdy - 2*uav.Jxz**2*uav.Jyz*wdz - uav.Jxy*uav.Jzz**2*wdx - uav.Jyy*uav.Jyz**2*wdy - 2*uav.Jyz**3*wdz - uav.Jyy*uav.Jzz**2*wdy + uav.Jyy**2*uav.Jzz*wdy + uav.Jyz**2*uav.Jzz*wdy - uav.Jxx*uav.Jxz*uav.Jyz*wdx + uav.Jxx*uav.Jxy*uav.Jzz*wdx + uav.Jxz*uav.Jyy*uav.Jyz*wdx + 2*uav.Jxy*uav.Jxz*uav.Jzz*wdz + uav.Jxy*uav.Jyy*uav.Jzz*wdx + uav.Jxz*uav.Jyz*uav.Jzz*wdx + 2*uav.Jyy*uav.Jyz*uav.Jzz*wdz)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[0, 12]  = -(2*wr1_ast*(uav.Jxy*uav.Jyz*uav.cd - uav.Jxz*uav.Jyy*uav.cd + uav.Jyz**2*uav.ct*uav.yl1 + uav.Jxz*uav.Jyz*uav.ct*uav.xl1 - uav.Jxy*uav.Jzz*uav.ct*uav.xl1 - uav.Jyy*uav.Jzz*uav.ct*uav.yl1))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[0, 13]  = -(2*wr2_ast*(uav.Jxy*uav.Jyz*uav.cd - uav.Jxz*uav.Jyy*uav.cd - uav.Jyz**2*uav.ct*uav.yl2 - uav.Jxz*uav.Jyz*uav.ct*uav.xl2 + uav.Jxy*uav.Jzz*uav.ct*uav.xl2 + uav.Jyy*uav.Jzz*uav.ct*uav.yl2))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[0, 14]  = (2*wr3_ast*(uav.Jxy*uav.Jyz*uav.cd - uav.Jxz*uav.Jyy*uav.cd + uav.Jyz**2*uav.ct*uav.yl3 - uav.Jxz*uav.Jyz*uav.ct*uav.xl3 + uav.Jxy*uav.Jzz*uav.ct*uav.xl3 - uav.Jyy*uav.Jzz*uav.ct*uav.yl3))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[0, 15]  = (2*wr4_ast*(uav.Jxy*uav.Jyz*uav.cd - uav.Jxz*uav.Jyy*uav.cd - uav.Jyz**2*uav.ct*uav.yl4 + uav.Jxz*uav.Jyz*uav.ct*uav.xl4 - uav.Jxy*uav.Jzz*uav.ct*uav.xl4 + uav.Jyy*uav.Jzz*uav.ct*uav.yl4))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[1, 0]   = -(uav.Jxx*uav.Jxz**2*wdz - 2*uav.Jxy**2*uav.Jxz*wdx - 2*uav.Jxz**3*wdx - uav.Jxx**2*uav.Jyz*wdy + uav.Jxx*uav.Jyz**2*wdz - 2*uav.Jxz**2*uav.Jyz*wdy + uav.Jxx*uav.Jzz**2*wdz - uav.Jxx**2*uav.Jzz*wdz - uav.Jxy**2*uav.Jzz*wdz - uav.Jxz**2*uav.Jzz*wdz + uav.Jxx*uav.Jxy*uav.Jxz*wdy + 2*uav.Jxx*uav.Jxy*uav.Jyz*wdx - uav.Jxy*uav.Jxz*uav.Jyy*wdy + 2*uav.Jxx*uav.Jxz*uav.Jzz*wdx + uav.Jxx*uav.Jyy*uav.Jyz*wdy + uav.Jxy*uav.Jxz*uav.Jzz*wdy + uav.Jxx*uav.Jyz*uav.Jzz*wdy)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[1, 1]   = -(2*uav.Jxy**2*uav.Jxz*wdy + 2*uav.Jxy*uav.Jxz**2*wdz - uav.Jxx**2*uav.Jyz*wdx - 2*uav.Jxz**2*uav.Jyz*wdx - 2*uav.Jxz*uav.Jyz**2*wdy + uav.Jxy*uav.Jzz**2*wdz + uav.Jxx*uav.Jxy*uav.Jxz*wdx - 2*uav.Jxx*uav.Jxy*uav.Jyz*wdy - uav.Jxy*uav.Jxz*uav.Jyy*wdx - uav.Jxx*uav.Jxz*uav.Jyz*wdz + uav.Jxx*uav.Jyy*uav.Jyz*wdx - uav.Jxx*uav.Jxy*uav.Jzz*wdz + uav.Jxy*uav.Jxz*uav.Jzz*wdx + uav.Jxz*uav.Jyy*uav.Jyz*wdz + uav.Jxx*uav.Jyz*uav.Jzz*wdx - uav.Jxy*uav.Jyy*uav.Jzz*wdz + 2*uav.Jxy*uav.Jyz*uav.Jzz*wdy - uav.Jxz*uav.Jyz*uav.Jzz*wdz)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[1, 2]   = (uav.Jxx**2*uav.Jzz*wdx - uav.Jxx*uav.Jxz**2*wdx - 2*uav.Jxy*uav.Jxz**2*wdy - uav.Jxx*uav.Jyz**2*wdx - 2*uav.Jxz*uav.Jyz**2*wdz - uav.Jxx*uav.Jzz**2*wdx - 2*uav.Jxz**3*wdz + uav.Jxy**2*uav.Jzz*wdx - uav.Jxy*uav.Jzz**2*wdy + uav.Jxz**2*uav.Jzz*wdx + uav.Jxx*uav.Jxz*uav.Jyz*wdy + uav.Jxx*uav.Jxy*uav.Jzz*wdy + 2*uav.Jxx*uav.Jxz*uav.Jzz*wdz - uav.Jxz*uav.Jyy*uav.Jyz*wdy + uav.Jxy*uav.Jyy*uav.Jzz*wdy + 2*uav.Jxy*uav.Jyz*uav.Jzz*wdz + uav.Jxz*uav.Jyz*uav.Jzz*wdy)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[1, 12]  = -(2*wr1_ast*(uav.Jxy*uav.Jxz*uav.cd - uav.Jxx*uav.Jyz*uav.cd - uav.Jxz**2*uav.ct*uav.xl1 + uav.Jxx*uav.Jzz*uav.ct*uav.xl1 - uav.Jxz*uav.Jyz*uav.ct*uav.yl1 + uav.Jxy*uav.Jzz*uav.ct*uav.yl1))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[1, 13]  = -(2*wr2_ast*(uav.Jxy*uav.Jxz*uav.cd - uav.Jxx*uav.Jyz*uav.cd + uav.Jxz**2*uav.ct*uav.xl2 - uav.Jxx*uav.Jzz*uav.ct*uav.xl2 + uav.Jxz*uav.Jyz*uav.ct*uav.yl2 - uav.Jxy*uav.Jzz*uav.ct*uav.yl2))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[1, 14]  = (2*wr3_ast*(uav.Jxy*uav.Jxz*uav.cd - uav.Jxx*uav.Jyz*uav.cd + uav.Jxz**2*uav.ct*uav.xl3 - uav.Jxx*uav.Jzz*uav.ct*uav.xl3 - uav.Jxz*uav.Jyz*uav.ct*uav.yl3 + uav.Jxy*uav.Jzz*uav.ct*uav.yl3))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[1, 15]  = (2*wr4_ast*(uav.Jxy*uav.Jxz*uav.cd - uav.Jxx*uav.Jyz*uav.cd - uav.Jxz**2*uav.ct*uav.xl4 + uav.Jxx*uav.Jzz*uav.ct*uav.xl4 + uav.Jxz*uav.Jyz*uav.ct*uav.yl4 - uav.Jxy*uav.Jzz*uav.ct*uav.yl4))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[2, 0]   = (uav.Jxx*uav.Jxy**2*wdy - 2*uav.Jxy**3*wdx - 2*uav.Jxy*uav.Jxz**2*wdx + uav.Jxx*uav.Jyy**2*wdy - uav.Jxx**2*uav.Jyy*wdy + uav.Jxx*uav.Jyz**2*wdy - uav.Jxy**2*uav.Jyy*wdy - uav.Jxx**2*uav.Jyz*wdz - uav.Jxz**2*uav.Jyy*wdy - 2*uav.Jxy**2*uav.Jyz*wdz + uav.Jxx*uav.Jxy*uav.Jxz*wdz + 2*uav.Jxx*uav.Jxy*uav.Jyy*wdx + 2*uav.Jxx*uav.Jxz*uav.Jyz*wdx + uav.Jxy*uav.Jxz*uav.Jyy*wdz + uav.Jxx*uav.Jyy*uav.Jyz*wdz - uav.Jxy*uav.Jxz*uav.Jzz*wdz + uav.Jxx*uav.Jyz*uav.Jzz*wdz)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[2, 1]   = -(uav.Jxx**2*uav.Jyy*wdx - uav.Jxx*uav.Jxy**2*wdx - 2*uav.Jxy**2*uav.Jxz*wdz - uav.Jxx*uav.Jyy**2*wdx - 2*uav.Jxy**3*wdy - uav.Jxx*uav.Jyz**2*wdx + uav.Jxy**2*uav.Jyy*wdx + uav.Jxz**2*uav.Jyy*wdx - 2*uav.Jxy*uav.Jyz**2*wdy - uav.Jxz*uav.Jyy**2*wdz + 2*uav.Jxx*uav.Jxy*uav.Jyy*wdy + uav.Jxx*uav.Jxy*uav.Jyz*wdz + uav.Jxx*uav.Jxz*uav.Jyy*wdz + uav.Jxy*uav.Jyy*uav.Jyz*wdz + 2*uav.Jxz*uav.Jyy*uav.Jyz*wdy - uav.Jxy*uav.Jyz*uav.Jzz*wdz + uav.Jxz*uav.Jyy*uav.Jzz*wdz)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[2, 2]   = (2*uav.Jxy**2*uav.Jxz*wdy + 2*uav.Jxy*uav.Jxz**2*wdz - uav.Jxx**2*uav.Jyz*wdx - 2*uav.Jxy**2*uav.Jyz*wdx + uav.Jxz*uav.Jyy**2*wdy - 2*uav.Jxy*uav.Jyz**2*wdz + uav.Jxx*uav.Jxy*uav.Jxz*wdx - uav.Jxx*uav.Jxy*uav.Jyz*wdy - uav.Jxx*uav.Jxz*uav.Jyy*wdy + uav.Jxy*uav.Jxz*uav.Jyy*wdx - 2*uav.Jxx*uav.Jxz*uav.Jyz*wdz + uav.Jxx*uav.Jyy*uav.Jyz*wdx - uav.Jxy*uav.Jxz*uav.Jzz*wdx - uav.Jxy*uav.Jyy*uav.Jyz*wdy + 2*uav.Jxz*uav.Jyy*uav.Jyz*wdz + uav.Jxx*uav.Jyz*uav.Jzz*wdx + uav.Jxy*uav.Jyz*uav.Jzz*wdy - uav.Jxz*uav.Jyy*uav.Jzz*wdy)/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[2, 12]  = (2*wr1_ast*(uav.Jxy**2*uav.cd - uav.Jxx*uav.Jyy*uav.cd - uav.Jxy*uav.Jxz*uav.ct*uav.xl1 + uav.Jxx*uav.Jyz*uav.ct*uav.xl1 + uav.Jxy*uav.Jyz*uav.ct*uav.yl1 - uav.Jxz*uav.Jyy*uav.ct*uav.yl1))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[2, 13]  = (2*wr2_ast*(uav.Jxy**2*uav.cd - uav.Jxx*uav.Jyy*uav.cd + uav.Jxy*uav.Jxz*uav.ct*uav.xl2 - uav.Jxx*uav.Jyz*uav.ct*uav.xl2 - uav.Jxy*uav.Jyz*uav.ct*uav.yl2 + uav.Jxz*uav.Jyy*uav.ct*uav.yl2))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[2, 14]  = -(2*wr3_ast*(uav.Jxy**2*uav.cd - uav.Jxx*uav.Jyy*uav.cd + uav.Jxy*uav.Jxz*uav.ct*uav.xl3 - uav.Jxx*uav.Jyz*uav.ct*uav.xl3 + uav.Jxy*uav.Jyz*uav.ct*uav.yl3 - uav.Jxz*uav.Jyy*uav.ct*uav.yl3))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[2, 15]  = -(2*wr4_ast*(uav.Jxy**2*uav.cd - uav.Jxx*uav.Jyy*uav.cd - uav.Jxy*uav.Jxz*uav.ct*uav.xl4 + uav.Jxx*uav.Jyz*uav.ct*uav.xl4 - uav.Jxy*uav.Jyz*uav.ct*uav.yl4 + uav.Jxz*uav.Jyy*uav.ct*uav.yl4))/(uav.Jxx*uav.Jyz**2 + uav.Jxz**2*uav.Jyy + uav.Jxy**2*uav.Jzz - 2*uav.Jxy*uav.Jxz*uav.Jyz - uav.Jxx*uav.Jyy*uav.Jzz) 
        A[3, 6]   = 1 
        A[4, 7]   = 1 
        A[5, 8]   = 1 
        A[6, 0]   = uav.wc**2 
        A[6, 3]   = -uav.wc**2 
        A[6, 6]   = -2**(1/2)*uav.wc 
        A[7, 1]   = uav.wc**2 
        A[7, 4]   = -uav.wc**2 
        A[7, 7]   = -2**(1/2)*uav.wc 
        A[8, 2]   = uav.wc**2 
        A[8, 5]   = -uav.wc**2 
        A[8, 8]   = -2**(1/2)*uav.wc 
        A[9, 0]   = -1 
        A[10, 1]  = -1 
        A[11, 2]  = -1 
        A[12, 0]  = uav.ar*uav.b11*uav.br*uav.kp_wx 
        A[12, 1]  = -uav.ar*uav.b12*uav.br*uav.kp_wy 
        A[12, 2]  = -uav.ar*uav.b13*uav.br*uav.kp_wz 
        A[12, 6]  = uav.ar*uav.b11*uav.br*uav.kd_wx 
        A[12, 7]  = -uav.ar*uav.b12*uav.br*uav.kd_wy 
        A[12, 8]  = -uav.ar*uav.b13*uav.br*uav.kd_wz 
        A[12, 9]  = -uav.ar*uav.b11*uav.br*uav.ki_wx 
        A[12, 10] = uav.ar*uav.b12*uav.br*uav.ki_wy 
        A[12, 11] = uav.ar*uav.b13*uav.br*uav.ki_wz 
        A[12, 12] = -uav.ar 
        A[13, 0]  = -uav.ar*uav.b21*uav.br*uav.kp_wx 
        A[13, 1]  = uav.ar*uav.b22*uav.br*uav.kp_wy 
        A[13, 2]  = -uav.ar*uav.b23*uav.br*uav.kp_wz 
        A[13, 6]  = -uav.ar*uav.b21*uav.br*uav.kd_wx 
        A[13, 7]  = uav.ar*uav.b22*uav.br*uav.kd_wy 
        A[13, 8]  = -uav.ar*uav.b23*uav.br*uav.kd_wz 
        A[13, 9]  = uav.ar*uav.b21*uav.br*uav.ki_wx 
        A[13, 10] = -uav.ar*uav.b22*uav.br*uav.ki_wy 
        A[13, 11] = uav.ar*uav.b23*uav.br*uav.ki_wz 
        A[13, 13] = -uav.ar 
        A[14, 0]  = -uav.ar*uav.b31*uav.br*uav.kp_wx 
        A[14, 1]  = -uav.ar*uav.b32*uav.br*uav.kp_wy 
        A[14, 2]  = uav.ar*uav.b33*uav.br*uav.kp_wz 
        A[14, 6]  = -uav.ar*uav.b31*uav.br*uav.kd_wx 
        A[14, 7]  = -uav.ar*uav.b32*uav.br*uav.kd_wy 
        A[14, 8]  = uav.ar*uav.b33*uav.br*uav.kd_wz 
        A[14, 9]  = uav.ar*uav.b31*uav.br*uav.ki_wx 
        A[14, 10] = uav.ar*uav.b32*uav.br*uav.ki_wy 
        A[14, 11] = -uav.ar*uav.b33*uav.br*uav.ki_wz 
        A[14, 14] = -uav.ar 
        A[15, 0]  = uav.ar*uav.b41*uav.br*uav.kp_wx 
        A[15, 1]  = uav.ar*uav.b42*uav.br*uav.kp_wy 
        A[15, 2]  = uav.ar*uav.b43*uav.br*uav.kp_wz 
        A[15, 6]  = uav.ar*uav.b41*uav.br*uav.kd_wx 
        A[15, 7]  = uav.ar*uav.b42*uav.br*uav.kd_wy 
        A[15, 8]  = uav.ar*uav.b43*uav.br*uav.kd_wz 
        A[15, 9]  = -uav.ar*uav.b41*uav.br*uav.ki_wx 
        A[15, 10] = -uav.ar*uav.b42*uav.br*uav.ki_wy 
        A[15, 11] = -uav.ar*uav.b43*uav.br*uav.ki_wz 
        A[15, 15] = -uav.ar 

        return A

def compute_A_level_two(uav, T):
        A1 = compute_A_level_one(uav, T, [0., 0., 0.])

        A_2_12 = np.hstack([
                        0.5 * np.eye(3),
                        np.zeros((3, 3)),
                        np.zeros((3, 3)),
                        np.zeros((3, 3)),
                        np.zeros((3, 4))
                ])
        
        Kpw = np.diag([uav.kp_wx, uav.kp_wy, uav.kp_wz])
        Bw = uav.B[:, 0:3]


        A_2_21 = np.vstack([
                        np.zeros((3,3)),
                        np.zeros((3,3)),
                        np.zeros((3,3)),
                        -uav.Kpq,
                        -uav.ar*uav.br*np.matmul(Bw, np.matmul(Kpw, uav.Kpq))
                ])
        
        A2 = np.block([
                        [np.zeros((3,3)), A_2_12],
                        [A_2_21,          A1    ]
                        ])
        return A2