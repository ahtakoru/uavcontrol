% EXAMPLE2  Stability analysis and nonlinear simulation of the Level-2
%           attitude control loop.
%
% PURPOSE
%   This script demonstrates how to:
%     1. Load default UAV parameters and override an attitude gain.
%     2. Validate the mixer and rotor geometry.
%     3. Linearise the Level-2 closed-loop dynamics and check stability.
%     4. Simulate the nonlinear system from a perturbed initial condition.
%     5. Plot the quaternion vector part response.
%
% See also: SETDEFAULTPARAMS, CHECKGEOMETRY, CHECKMIXER,
%           LINEARIZEDMATRICESLEVEL2, COMPUTELEVEL1EQPOINT, SIMULATELEVEL2

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

clear;  clc;

%% --- Load default parameters and override attitude gain -----------------
uav        = setDefaultParams();
uav.kp_q3  = 155.0;
uav.Kpq    = diag([uav.kp_q1, uav.kp_q2, uav.kp_q3]);

%% --- Validate geometry and mixer ----------------------------------------
checkGeometry(uav);
checkMixer(uav);

%% --- Operating point ----------------------------------------------------
T  = 0.5;             % collective thrust  (N)
qd = [1; 0; 0; 0];   % desired attitude quaternion  (identity → hover)

%% --- Linearisation and stability analysis -------------------------------
[~, ~, ~, ~, ~, ~, Ac1c2] = linearizedMatricesLevel2(uav, T);

eigvals     = eig(Ac1c2);
maxRealPart = max(real(eigvals));

fprintf('  Closed-loop eigenvalue analysis:\n');
fprintf('    max(Re(eig(Ac1c2))) = %.6f\n\n', maxRealPart);

if maxRealPart < 0
    fprintf('  [PASS] Equilibrium is exponentially stable.\n\n');
else
    fprintf('  [FAIL] Equilibrium is NOT exponentially stable.\n\n');
end

%% --- Compute equilibrium state for initialisation -----------------------
x1_star = computeLevel1EqPoint(uav, zeros(3,1), T);

%% --- Initial condition: small quaternion perturbation around hover ------
x2_init           = zeros(26, 1);
x2_init(7:10)     = qd + [0.05; 0.05; 0.05; 0.05];
x2_init(7:10)     = x2_init(7:10) / norm(x2_init(7:10));
x2_init(11:end)   = x1_star;

%% --- Simulate nonlinear dynamics ----------------------------------------
tspan = [0, 10];   % simulation horizon  (s)
[t, p, v, q, w, wr, z, wf, alpha] = simulateLevel2(uav, tspan, x2_init, qd, T);

%% --- Plot quaternion vector part response --------------------------------
clf;
plot(t, q(:, 2:4), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Quaternion vector part');
title('Level-2 Attitude Response — Quaternion Vector Part');
legend('q_1', 'q_2', 'q_3', 'Location', 'best');
grid on;