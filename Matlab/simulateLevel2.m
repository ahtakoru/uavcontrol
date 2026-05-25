function [t, p, v, q, w, wr, z, wf, alpha] = simulateLevel2(uav, tspan, x2_init, qd, T)
% SIMULATELEVEL2  Simulate the full nonlinear Level-2 attitude control loop.
%
%   [t, p, v, q, w, wr, z, wf, alpha] = simulateLevel2(uav, tspan, x2_init, qd, T)
%
%   Integrates the 26-state nonlinear UAV equations of motion under the
%   two-level cascade controller (Level-2 attitude + Level-1 angular rate)
%   using ode45.
%
% STATE VECTOR  x2 (26×1)
%   x2 = [ p      (3×1)  position                       (m)     ]
%        [ v      (3×1)  velocity                        (m/s)   ]
%        [ q      (4×1)  attitude quaternion             (—)     ]
%        [ w      (3×1)  body angular rate               (rad/s) ]
%        [ wr     (4×1)  rotor speeds                    (rad/s) ]
%        [ z      (3×1)  angular-rate integrator         (rad)   ]
%        [ wf     (3×1)  filtered angular rate            (rad/s)]
%        [ alpha  (3×1)  angular acceleration            (rad/s²)]
%
% INPUT
%   uav      - UAV parameter struct (see SETDEFAULTPARAMS)
%   tspan    - (1×2) time interval  [t0, tf]  (s)
%   x2_init  - (26×1) initial state vector
%   qd       - (4×1) desired attitude quaternion  [q0; q1; q2; q3]
%   T        - collective thrust command  (N)
%
% OUTPUT
%   t      - (N×1)  time vector                (s)
%   p      - (N×3)  position trajectory        (m)
%   v      - (N×3)  velocity trajectory        (m/s)
%   q      - (N×4)  quaternion trajectory      (—,  re-normalised)
%   w      - (N×3)  angular rate trajectory    (rad/s)
%   wr     - (N×4)  rotor speed trajectory     (rad/s)
%   z      - (N×3)  integrator trajectory      (rad)
%   wf     - (N×3)  filtered rate trajectory   (rad/s)
%   alpha  - (N×3)  acceleration trajectory    (rad/s²)
%
% EXAMPLE
%   uav     = setDefaultParams();
%   qd      = [1; 0; 0; 0];
%   x2_init = [zeros(6,1); qd; zeros(16,1)];
%   [t, p, v, q, w, wr, z, wf, alpha] = simulateLevel2(uav, [0 10], x2_init, qd, 0.5);
%
% See also: SETDEFAULTPARAMS, LINEARIZEDMATRICESLEVEL2, QUATERNIONTOOLS

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    %% --- Normalise initial quaternion ------------------------------------
    x2_init(7:10) = x2_init(7:10) / norm(x2_init(7:10));

    %% --- Integrate equations of motion -----------------------------------
    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
    [t, x2] = ode45(@(t, x) quadODE(t, x, uav, qd, T), tspan, x2_init, opts);

    %% --- Unpack state trajectories ---------------------------------------
    p     = x2(:,  1:3);
    v     = x2(:,  4:6);
    q     = x2(:,  7:10);
    w     = x2(:, 11:13);
    wr    = x2(:, 14:17);
    z     = x2(:, 18:20);
    wf    = x2(:, 21:23);
    alpha = x2(:, 24:26);

    %% --- Re-normalize quaternion (correct ODE integration drift) ---------
    q = q ./ vecnorm(q, 2, 2);

end


% ==========================================================================
function dx = quadODE(t, x, uav, qd, T) %#ok<INUSL>
% QUADODE  Equations of motion for the Level-2 closed-loop UAV system.
%
%   dx = quadODE(t, x, uav, qd, T)
%
%   Called internally by SIMULATELEVEL2 via ode45. The time argument t is
%   unused (autonomous system) but required by the ode45 interface.
%
% CONTROL LAW
%   Level 2:  wd  = −Kpq · sign(qe0) · qe_vec
%   Level 1:  tau = −Kpw·(w − wd) − Kiw·z − Kdw·alpha

    %% --- Unpack state ---------------------------------------------------
    p     = x(1:3);    %#ok<NASGU>  position (unused in rotation dynamics)
    v     = x(4:6);
    q     = x(7:10)  / norm(x(7:10));   % normalise at each ODE step
    w     = x(11:13);
    wr    = x(14:17);
    z     = x(18:20);
    wf    = x(21:23);
    alpha = x(24:26);

    g = 9.81;   % gravitational acceleration  (m/s²)

    %% --- Level-2: attitude error and rate reference ---------------------
    qe = QuaternionTools.multiplication(QuaternionTools.inverse(qd), q);
    wd = -uav.Kpq * QuaternionTools.sign(qe(1)) * QuaternionTools.imag(qe);
    wd = min(max(wd, -uav.wdmax), uav.wdmax); % Clip between -wdmax, wdmax

    %% --- Level-1: angular rate PID control law --------------------------
    tau_x = -uav.kp_wx*(w(1) - wd(1)) - uav.ki_wx*z(1) - uav.kd_wx*alpha(1);
    tau_y = -uav.kp_wy*(w(2) - wd(2)) - uav.ki_wy*z(2) - uav.kd_wy*alpha(2);
    tau_z = -uav.kp_wz*(w(3) - wd(3)) - uav.ki_wz*z(3) - uav.kd_wz*alpha(3);

    PWM = uav.M * [tau_x; tau_y; tau_z; T];

    %% --- Aerodynamic forces and torques ---------------------------------
    torqueForce = uav.G * wr.^2;
    tau         = torqueForce(1:3);
    f           = torqueForce(4);

    %% --- Nonlinear equations of motion ----------------------------------
    dp     = v;
    dv     = -QuaternionTools.rotationMatrix(q) * f * [0; 0; 1] + uav.m * g * [0; 0; 1];
    dq     = 0.5 * QuaternionTools.multiplication(q, QuaternionTools.embed(w));
    dw     = uav.J \ (cross(-w, uav.J*w) + tau);
    dwr    = -uav.ar * wr + uav.br * PWM;
    dz     = w - wd;
    dwf    = alpha;
    dalpha = -uav.wc^2 * wf - sqrt(2) * uav.wc * alpha + uav.wc^2 * w;

    dx = [dp; dv; dq; dw; dwr; dz; dwf; dalpha];

end