import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import quaternion_tools as qt

def simulate_level_1(uav, T, wd, 
                     w0 = np.zeros(3), 
                     alpha0 = np.zeros(3),
                     z0 = np.zeros(3),
                     wr0 = np.zeros(4), 
                     tspan = np.arange(0, 10, 0.05)):
    
    q0 = np.concatenate([w0, w0, alpha0, z0, wr0])
    
    def ode_level1(t, q):
        dq = np.zeros(16)

        # Fetch states
        wx,  wy,  wz           = q[0],  q[1],  q[2]
        wfx, wfy, wfz          = q[3],  q[4],  q[5]
        alphax, alphay, alphaz = q[6],  q[7],  q[8]
        zx, zy, zz             = q[9],  q[10], q[11]
        wr1, wr2, wr3, wr4     = q[12], q[13], q[14], q[15]
        wdx, wdy, wdz = wd[0], wd[1], wd[2]

        # Compact forms
        w = np.array([wx, wy, wz])
        wf = np.array([wfx, wfy, wfz])
        alpha = np.array([alphax, alphay, alphaz])

        wr = np.array([wr1, wr2, wr3, wr4])
        
        torques_forces = np.matmul(uav.G, wr**2)
        tau = torques_forces[0:3]
        dw = np.matmul(uav.Jinv, -np.cross(w, np.matmul(uav.J, w))) + np.matmul(uav.Jinv, tau)
        dwf = alpha
        dalpha = -uav.wc**2*wf - np.sqrt(2)*uav.wc*alpha + uav.wc**2*w
        dz = np.array([wdx - wx, wdy - wy, wdz - wz])

        # Controls Part
        taudx = uav.kp_wx*(wdx - wx) + uav.ki_wx*zx - uav.kd_wx*alphax
        taudy = uav.kp_wy*(wdy - wy) + uav.ki_wy*zy - uav.kd_wy*alphay
        taudz = uav.kp_wz*(wdz - wz) + uav.ki_wz*zz - uav.kd_wz*alphaz
        
        taud = np.array([taudx, taudy, taudz])

        PWM = np.matmul(uav.B, np.concatenate([taud, np.array([T])]))
        dwr = -uav.ar*(wr - uav.br*PWM)

        for i in range(4):
            if wr[i] <= 0.: dwr[i] = 0

        dq = np.concatenate([dw, dwf, dalpha, dz, dwr])
        return dq
    
    sol = solve_ivp(ode_level1, t_span=(tspan[0], tspan[-1]), y0=q0, t_eval=tspan, method="RK45")
    return sol

def simulate_level_2(uav, T, qd, 
                     quat0 = np.array([1., 0., 0., 0.]),
                     w0 = np.zeros(3),
                     alpha0 = np.zeros(3), 
                     z0 = np.zeros(3),
                     wr0 = np.zeros(4), 
                     tspan = np.arange(0, 20, 0.05)):
    
    q0 = np.concatenate([quat0, w0, w0, alpha0, z0, wr0])
    
    def ode_level2(t, q):
        dq = np.zeros(20)

        # Fetch states
        quat_r, quat_x, quat_y, quat_z = q[0], q[1], q[2], q[3]
        wx,  wy,  wz                   = q[4],  q[5],  q[6]
        wfx, wfy, wfz                  = q[7],  q[8],  q[9]
        alphax, alphay, alphaz         = q[10], q[11], q[12]
        zx, zy, zz                     = q[13], q[14], q[15]
        wr1, wr2, wr3, wr4             = q[16], q[17], q[18], q[19]

        
        # Compact forms
        quat = np.array([quat_r, quat_x, quat_y, quat_z])
        quat_err = qt.normalize(qt.multiply(qt.inverse(quat), qd))
        w = np.array([wx, wy, wz])
        wf = np.array([wfx, wfy, wfz])
        alpha = np.array([alphax, alphay, alphaz])

        wd = qt.sign(quat_err[0])*np.matmul(uav.Kpq, qt.imag(quat_err))
        wdx, wdy, wdz = wd[0], wd[1], wd[2]

        wr = np.array([wr1, wr2, wr3, wr4])
        
        torques_forces = np.matmul(uav.G, wr**2)
        tau = torques_forces[0:3]

        dquat = 1/2*qt.multiply(quat, qt.embed_vec(w))
        
        dw = np.matmul(uav.Jinv, -np.cross(w, np.matmul(uav.J, w))) + np.matmul(uav.Jinv, tau)
        dwf = alpha
        dalpha = -uav.wc**2*wf - np.sqrt(2)*uav.wc*alpha + uav.wc**2*w
        dz = np.array([wdx - wx, wdy - wy, wdz - wz])

        # Controls Part
        taudx = uav.kp_wx*(wdx - wx) + uav.ki_wx*zx - uav.kd_wx*alphax
        taudy = uav.kp_wy*(wdy - wy) + uav.ki_wy*zy - uav.kd_wy*alphay
        taudz = uav.kp_wz*(wdz - wz) + uav.ki_wz*zz - uav.kd_wz*alphaz
        
        taud = np.array([taudx, taudy, taudz])

        PWM = np.matmul(uav.B, np.concatenate([taud, np.array([T])]))
        dwr = -uav.ar*(wr - uav.br*PWM)


        dq = np.concatenate([dquat, dw, dwf, dalpha, dz, dwr])
        return dq
    
    sol = solve_ivp(ode_level2, t_span=(tspan[0], tspan[-1]), y0=q0, t_eval=tspan, method="RK45")
    return sol

def plot_results_level_1(sol, wd, to_deg=False, save_path=None, show=True):
    """
    Plot Level-1 simulation results from solve_ivp solution.

    Args:
        sol: solve_ivp result (expects 16-state y with shape (16, N))
        to_deg: if True, plot angular rates in deg/s
        save_path: if set (e.g., "figs/level1.png"), saves the full figure
        show: whether to call plt.show() at the end

    Returns:
        fig: the matplotlib Figure
        axes: dict of axes for convenience
    """
    t = sol.t
    Y = sol.y  # shape (16, N)

    w   = Y[0:3, :]      # body rates
    wf  = Y[3:6, :]      # filtered body rates
    alp = Y[6:9, :]      # alpha (filter states)
    z   = Y[9:12, :]     # integral states
    wr  = Y[12:16, :]    # rotor speeds

    conv = 180.0/np.pi if to_deg else 1.0
    rate_unit = "deg/s" if to_deg else "rad/s"

    fig = plt.figure(figsize=(12, 10))
    axes = {}

    # 1) Body rates vs filtered rates
    ax11 = plt.subplot(3, 4, 1)
    ax11.plot(t, conv*wd[0]*np.ones_like(t), "--k", label="wdx")
    ax11.plot(t, conv*w[0], label="wx")
    # ax11.plot(t, conv*wf[0], "--", label="wfx")
    ax11.set_title("x-axis angular rate and filtered rate")
    ax11.set_xlabel("Time [s]")
    ax11.set_ylabel(f"Rate [{rate_unit}]")
    ax11.grid(True, alpha=0.3)
    ax11.legend(ncol=1)

    ax12 = plt.subplot(3, 4, 2)
    ax12.plot(t, conv*wd[1]*np.ones_like(t), "--k", label="wdy")
    ax12.plot(t, conv*w[1], label="wy")
    ax12.plot(t, conv*wf[1], "--", label="wfy")
    ax12.set_title("y-axis angular rate and filtered rate")
    ax12.set_xlabel("Time [s]")
    ax12.set_ylabel(f"Rate [{rate_unit}]")
    ax12.grid(True, alpha=0.3)
    ax12.legend(ncol=1)

    ax13 = plt.subplot(3, 4, 3)
    ax13.plot(t, conv*wd[2]*np.ones_like(t), "--k", label="wdz")
    ax13.plot(t, conv*w[2], label="wz")
    ax13.plot(t, conv*wf[2], "--", label="wfz")
    ax13.set_title("z-axis angular rate and filtered rate")
    ax13.set_xlabel("Time [s]")
    ax13.set_ylabel(f"Rate [{rate_unit}]")
    ax13.grid(True, alpha=0.3)
    ax13.legend(ncol=1)

    # 2) wrotor
    ax21 = plt.subplot(3, 4, 5)
    ax21.plot(t, wr[0], label="wr1")
    ax21.set_title("Rotor Speed 1")
    ax21.set_xlabel("Time [s]")
    ax21.set_ylabel(f"Speed [RPM]")
    ax21.grid(True, alpha=0.3)

    ax22 = plt.subplot(3, 4, 6)
    ax22.plot(t, wr[1], label="wr2")
    ax22.set_title("Rotor Speed 2")
    ax22.set_xlabel("Time [s]")
    ax22.set_ylabel(f"Speed [RPM]")
    ax22.grid(True, alpha=0.3)

    ax23 = plt.subplot(3, 4, 7)
    ax23.plot(t, wr[2], label="wr3")
    ax23.set_title("Rotor Speed 3")
    ax23.set_xlabel("Time [s]")
    ax23.set_ylabel(f"Speed [RPM]")
    ax23.grid(True, alpha=0.3)

    ax24 = plt.subplot(3, 4, 8)
    ax24.plot(t, wr[3], label="wr4")
    ax24.set_title("Rotor Speed 4")
    ax24.set_xlabel("Time [s]")
    ax24.set_ylabel(f"Speed [RPM]")
    ax24.grid(True, alpha=0.3)

    # 3) z-states
    ax31 = plt.subplot(3, 4, 9)
    ax31.plot(t, z[0], label="zx")
    ax31.set_title("x-axis Integral State")
    ax31.set_xlabel("Time [s]")
    ax31.set_ylabel(f"zx [{rate_unit}]")
    ax31.grid(True, alpha=0.3)

    ax32 = plt.subplot(3, 4, 10)
    ax32.plot(t, z[1], label="zy")
    ax32.set_title("y-axis Integral State")
    ax32.set_xlabel("Time [s]")
    ax32.set_ylabel(f"zy [{rate_unit}]")
    ax32.grid(True, alpha=0.3)

    ax33 = plt.subplot(3, 4, 11)
    ax33.plot(t, z[2], label="zz")
    ax33.set_title("z-axis Integral State")
    ax33.set_xlabel("Time [s]")
    ax33.set_ylabel(f"zz [{rate_unit}]")
    ax33.grid(True, alpha=0.3)    

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes

def plot_results_level_2(sol, qd, to_deg=False, save_path=None, show=True):
    """
    Plot Level-2 simulation results from solve_ivp solution.

    Args:
        sol: solve_ivp result (expects 16-state y with shape (16, N))
        to_deg: if True, plot angular rates in deg/s
        save_path: if set (e.g., "figs/level1.png"), saves the full figure
        show: whether to call plt.show() at the end

    Returns:
        fig: the matplotlib Figure
        axes: dict of axes for convenience
    """
    t = sol.t
    Y = sol.y  # shape (20, N)

    quat = Y[0:4, :]      # body rates
    w    = Y[4:7, :]      # body rates
    wf   = Y[7:10, :]      # filtered body rates
    alp  = Y[10:13, :]      # alpha (filter states)
    z    = Y[13:16, :]     # integral states
    wr   = Y[16:20, :]    # rotor speeds

    conv = 180.0/np.pi if to_deg else 1.0
    rate_unit = "deg/s" if to_deg else "rad/s"

    fig = plt.figure(figsize=(16, 10))
    axes = {}

    # 1) Quaternions
    ax11 = plt.subplot(4, 4, 1)
    ax11.plot(t, np.ones_like(t)*qd[0], "--k", label="qd0")
    ax11.plot(t, quat[0], label="q0")
    ax11.set_title("Quaternion")
    ax11.set_xlabel("Time [s]")
    ax11.set_ylabel("q_0")
    ax11.grid(True, alpha=0.3)
    ax11.legend(ncol=1)

    ax12 = plt.subplot(4, 4, 2)
    ax12.plot(t, np.ones_like(t)*qd[1], "--k", label="qd1")
    ax12.plot(t, quat[1], label="q1")
    ax12.set_title("Quaternion")
    ax12.set_xlabel("Time [s]")
    ax12.set_ylabel("q_1")
    ax12.grid(True, alpha=0.3)
    ax12.legend(ncol=1)

    ax13 = plt.subplot(4, 4, 3)
    ax13.plot(t, np.ones_like(t)*qd[2], "--k", label="qd2")
    ax13.plot(t, quat[2], label="q2")
    ax13.set_title("xQuaternion")
    ax13.set_xlabel("Time [s]")
    ax13.set_ylabel("q_2")
    ax13.grid(True, alpha=0.3)
    ax13.legend(ncol=1)

    ax14 = plt.subplot(4, 4, 4)
    ax14.plot(t, np.ones_like(t)*qd[3], "--k", label="qd3")
    ax14.plot(t, quat[3], label="q3")
    ax14.set_title("Quaternion")
    ax14.set_xlabel("Time [s]")
    ax14.set_ylabel("q_3")
    ax14.grid(True, alpha=0.3)
    ax14.legend(ncol=1)

    # 2) Body rates vs filtered rates
    ax21 = plt.subplot(4, 4, 5)
    ax21.plot(t, conv*w[0], label="wx")
    ax21.plot(t, conv*wf[0], "--", label="wfx")
    ax21.set_title("x-axis angular rate and filtered rate")
    ax21.set_xlabel("Time [s]")
    ax21.set_ylabel(f"Rate [{rate_unit}]")
    ax21.grid(True, alpha=0.3)
    ax21.legend(ncol=1)

    ax22 = plt.subplot(4, 4, 6)
    ax22.plot(t, conv*w[1], label="wy")
    ax22.plot(t, conv*wf[1], "--", label="wfy")
    ax22.set_title("y-axis angular rate and filtered rate")
    ax22.set_xlabel("Time [s]")
    ax22.set_ylabel(f"Rate [{rate_unit}]")
    ax22.grid(True, alpha=0.3)
    ax22.legend(ncol=1)

    ax23 = plt.subplot(4, 4, 7)
    ax23.plot(t, conv*w[2], label="wz")
    ax23.plot(t, conv*wf[2], "--", label="wfz")
    ax23.set_title("z-axis angular rate and filtered rate")
    ax23.set_xlabel("Time [s]")
    ax23.set_ylabel(f"Rate [{rate_unit}]")
    ax23.grid(True, alpha=0.3)
    ax23.legend(ncol=1)

    # 3) wrotor
    ax31 = plt.subplot(4, 4, 9)
    ax31.plot(t, wr[0], label="wr1")
    ax31.set_title("Rotor Speed 1")
    ax31.set_xlabel("Time [s]")
    ax31.set_ylabel(f"Speed [RPM]")
    ax31.grid(True, alpha=0.3)

    ax32 = plt.subplot(4, 4, 10)
    ax32.plot(t, wr[1], label="wr2")
    ax32.set_title("Rotor Speed 2")
    ax32.set_xlabel("Time [s]")
    ax32.set_ylabel(f"Speed [RPM]")
    ax32.grid(True, alpha=0.3)

    ax33 = plt.subplot(4, 4, 11)
    ax33.plot(t, wr[2], label="wr3")
    ax33.set_title("Rotor Speed 3")
    ax33.set_xlabel("Time [s]")
    ax33.set_ylabel(f"Speed [RPM]")
    ax33.grid(True, alpha=0.3)

    ax44 = plt.subplot(4, 4, 12)
    ax44.plot(t, wr[3], label="wr4")
    ax44.set_title("Rotor Speed 4")
    ax44.set_xlabel("Time [s]")
    ax44.set_ylabel(f"Speed [RPM]")
    ax44.grid(True, alpha=0.3)

    # 4) z-states
    ax51 = plt.subplot(4, 4, 13)
    ax51.plot(t, z[0], label="zx")
    ax51.set_title("x-axis Integral State")
    ax51.set_xlabel("Time [s]")
    ax51.set_ylabel(f"zx [{rate_unit}]")
    ax51.grid(True, alpha=0.3)

    ax52 = plt.subplot(4, 4, 14)
    ax52.plot(t, z[1], label="zy")
    ax52.set_title("y-axis Integral State")
    ax52.set_xlabel("Time [s]")
    ax52.set_ylabel(f"zy [{rate_unit}]")
    ax52.grid(True, alpha=0.3)

    ax53 = plt.subplot(4, 4, 15)
    ax53.plot(t, z[2], label="zz")
    ax53.set_title("z-axis Integral State")
    ax53.set_xlabel("Time [s]")
    ax53.set_ylabel(f"zz [{rate_unit}]")
    ax53.grid(True, alpha=0.3)    

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes

def save_level1_longdat(sol, filepath="level1_results_long.dat"):
    """
    Save simulation results as a wide matrix: one row per time point.

    Columns (no header): 
      t, wx, wy, wz, wfx, wfy, wfz,
      alphax, alphay, alphaz, zx, zy, zz, wr1, wr2, wr3, wr4
    """
    t = np.array(sol.t)      # (N,)
    Y = np.array(sol.y)      # (16, N)

    R = np.column_stack([t, Y.T])   # (N, 17)

    np.savetxt(filepath, R, fmt="%.10g", delimiter=" ")
    return R