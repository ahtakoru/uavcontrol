function [x1_star, w_star, wr_star, z_star, wf_star, alpha_star, f_star] = ...
         computeLevel1EqPoint(uav, wd, T)
%   Compute the equilibrium point of the UAV angular-rate loop.
%
%   [x1_star, w_star, wr_star, z_star, wf_star, alpha_star, f_star] = ...
%       compute_eq_point(uav, T, wd)
%
%   Solves for the steady-state rotor speeds (wr_star) and integrator states
%   (z_star) using a Newton-Raphson iteration, then evaluates the full
%   nonlinear dynamics to verify that f_star ≈ 0 at the equilibrium.
%
% INPUTS
%   uav  - UAV parameter struct with fields:
%            .J        (3×3) inertia matrix
%            .Gtau     (3×4) torque-allocation matrix
%            .M        (4×4) control-mixing matrix
%            .M_T      (4×1) thrust-coefficient vector  (or scalar)
%            .M_w      (4×4) rotor-speed mixing matrix
%            .ar, .br  rotor first-order model coefficients (scalars)
%            .Kiw      (4×3) integral-gain matrix
%            .kp_wx/wy/wz, ki_wx/wy/wz, kd_wx/wy/wz  PID gains (scalars)
%            .wc       Butterworth filter cut-off frequency (rad/s)
%   wd   - (3×1) desired angular velocity at equilibrium (rad/s)
%   T    - collective thrust command at equilibrium (N)
%
% OUTPUTS
%   x1_star    - (16×1) full equilibrium state vector [w; wr; z; wf; alpha]
%   w_star     - (3×1) equilibrium body angular rate  (= wd)
%   wr_star    - (4×1) equilibrium rotor speeds (rad/s)
%   z_star     - (3×1) equilibrium integrator states
%   wf_star    - (3×1) equilibrium filtered angular rate  (= wd)
%   alpha_star - (3×1) equilibrium angular acceleration  (= 0)
%   f_star     - (16×1) dynamics residual at equilibrium (should be ≈ 0)
%
% ALGORITHM
%   Newton-Raphson iteration on the coupled system
%       Gtau * wr² = cross(wd, J*wd)          (torque balance)
%       ar*wr + br*M_w*Kiw*z = br*M_T*T       (rotor steady-state)
%   convergence criterion: ||J⁻¹ F||₂ < tol
%
%
% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    %% --- Parameters -------------------------------------------------------
    TOL = 1e-9;       % Newton-Raphson convergence tolerance
    MAX_ITER = 200;   % safeguard against infinite loops

    %% --- Initial guess for rotor speeds -----------------------------------
    if all(uav.M_T > 0)
        wr_star = (uav.br / uav.ar) * T * uav.M_T;
    else
        % Fallback: distribute thrust equally over all rotors
        wr_star = (uav.br / uav.ar) * T * sum(uav.M_T) * ones(4, 1);
    end

    z_star = ones(3, 1);   % integrator state initial guess

    %% --- Newton-Raphson iteration -----------------------------------------
    for iter = 1:MAX_ITER
        % Jacobian of F w.r.t. [wr; z]
        J_nr = [ 2 * uav.Gtau * diag(wr_star),   zeros(3, 3)    ; ...
                    -uav.ar * eye(4),             -uav.br * uav.M_w * uav.Kiw ];

        % Residual vector
        F = [ uav.Gtau * wr_star.^2 - cross(wd, uav.J * wd)                       ; ...
             -uav.ar * wr_star - uav.br * uav.M_w * uav.Kiw * z_star + uav.br * uav.M_T * T ];

        % Newton step  (use \ for numerical stability instead of inv)
        delta = J_nr \ F;

        wr_star = wr_star - delta(1:4);
        z_star  = z_star  - delta(5:end);

        if norm(delta, 2) < TOL
            break
        end

        if iter == MAX_ITER
            warning('compute_eq_point:noConvergence', ...
                'Newton-Raphson did not converge in %d iterations (residual = %.3e).', ...
                MAX_ITER, norm(delta, 2));
        end
    end

    %% --- Remaining equilibrium states ------------------------------------
    w_star     = wd;
    wf_star    = wd;
    alpha_star = zeros(3, 1);

    %% --- Equilibrium torque demands and PWM commands ---------------------
    wdx = wd(1);  wdy = wd(2);  wdz = wd(3);

    tau_x = -uav.kp_wx * (w_star(1) - wdx) - uav.ki_wx * z_star(1) - uav.kd_wx * alpha_star(1);
    tau_y = -uav.kp_wy * (w_star(2) - wdy) - uav.ki_wy * z_star(2) - uav.kd_wy * alpha_star(2);
    tau_z = -uav.kp_wz * (w_star(3) - wdz) - uav.ki_wz * z_star(3) - uav.kd_wz * alpha_star(3);

    PWM = uav.M * [tau_x; tau_y; tau_z; T];

    %% --- Nonlinear dynamics residual (should be ≈ 0 at equilibrium) -----
    dw     = uav.J \ (cross(-w_star, uav.J * w_star) + uav.Gtau * wr_star.^2);
    dwr    = -uav.ar * wr_star + uav.br * PWM;
    dz     = w_star - wd;
    dwf    = alpha_star;
    dalpha = -uav.wc^2 * wf_star - sqrt(2) * uav.wc * alpha_star + uav.wc^2 * w_star;  % 2nd-order Butterworth

    %% --- Pack outputs ----------------------------------------------------
    f_star  = [dw; dwr; dz; dwf; dalpha];
    x1_star = [w_star; wr_star; z_star; wf_star; alpha_star];
    
    %% --- Convergence and admissibility report ----------------------------
    converged      = norm(f_star, 1) < 1e-6;
    admissible     = all(wr_star >= 0);

    if converged && admissible
        fprintf('  [PASS] Equilibrium found — physically admissible (all rotor speeds non-negative).\n\n');
    elseif converged && ~admissible
        fprintf('  [WARN] Equilibrium found — but rotor speeds contain negative values.\n');
        fprintf('         Check thrust level T = %.4f or mixer matrix.\n\n', T);
    else
        fprintf('  [FAIL] Newton-Raphson did not converge (residual = %.3e).\n\n', norm(f_star, 1));
    end

end