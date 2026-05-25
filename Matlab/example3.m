% EXAMPLE3  Sequential two-level controller design using the iLMI algorithm
%           followed by nonlinear simulation and response visualization.
%
% PURPOSE
%   This script demonstrates how to:
%     1. Design the Level-1 angular-rate controller via iLMI (inner loop).
%     2. Update the UAV parameter struct with the designed gains.
%     3. Design the Level-2 attitude controller via iLMI (outer loop).
%     4. Simulate the nonlinear closed-loop system from hover to a target attitude.
%     5. Plot the quaternion and angular rate responses.
%
% CONTROLLER DESIGN APPROACH
%   The two loops are designed sequentially:
%     - Inner loop (Level 1): minimizes alpha for the angular-rate dynamics
%         Ac1 = Ao1 - B1*K1*C1,  K1 = [diag(Kp), diag(Ki), diag(Kd)]
%     - Outer loop (Level 2): minimizes alpha for the attitude dynamics
%         Ac1c2 = Ac1o2 - Bc1o2*Kpq*Cc1o2,  Kpq = diag(kp_q1, kp_q2, kp_q3)
%
% DEPENDENCIES
%   YALMIP (sdpvar, optimize, sdpsettings, value)
%   Control System Toolbox (icare)
%
% See also: SETDEFAULTPARAMS, LINEARIZEDMATRICESLEVEL1, LINEARIZEDMATRICESLEVEL2,
%           DESIGNLEVEL1CONTROLLERILMI, DESIGNLEVEL2CONTROLLERILMI, SIMULATELEVEL2

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

clear;  clc;

%% --- Load default parameters --------------------------------------------
uav = setDefaultParams();

%% --- Operating point ----------------------------------------------------
wd = zeros(3, 1);   % desired angular rate at equilibrium  (rad/s)
T  = 0.5;           % collective thrust                    (N)

%% --- Level-1: inner loop design -----------------------------------------
fprintf('=== Level-1 Controller Design (Inner Loop) ===\n');

[Ao1, B1, C1, ~, ~] = linearizedMatricesLevel1(uav, wd, T);

Q1 = blkdiag(0.1   * eye(3), ...   % w      penalty
             1e-6  * eye(4), ...   % wr     penalty
             0.005 * eye(3), ...   % z      penalty
             1e-6  * eye(3), ...   % wf     penalty
             1e-6  * eye(3));      % alpha  penalty

K1 = designLevel1ControllerILMI(Ao1, -B1, C1, Q1, true);

fprintf('\n  Designed K1:\n');
disp(K1);

%% --- Update UAV struct with designed Level-1 gains ----------------------
uav.kp_wx = K1(1,1);  uav.kp_wy = K1(2,2);  uav.kp_wz = K1(3,3);
uav.ki_wx = K1(1,4);  uav.ki_wy = K1(2,5);  uav.ki_wz = K1(3,6);
uav.kd_wx = K1(1,7);  uav.kd_wy = K1(2,8);  uav.kd_wz = K1(3,9);

uav.Kpw = diag([uav.kp_wx, uav.kp_wy, uav.kp_wz]);
uav.Kiw = diag([uav.ki_wx, uav.ki_wy, uav.ki_wz]);
uav.Kdw = diag([uav.kd_wx, uav.kd_wy, uav.kd_wz]);

%% --- Level-2: outer loop design -----------------------------------------
fprintf('=== Level-2 Controller Design (Outer Loop) ===\n');

[~, ~, ~, Ac1o2, Bc1o2, Cc1o2, ~] = linearizedMatricesLevel2(uav, T);

Q2 = blkdiag(1      * eye(3), ...   % qe_vec  penalty
             0.1    * eye(3), ...   % w       penalty
             0.0001 * eye(4), ...   % wr      penalty
             0.0001 * eye(3), ...   % z       penalty
             1e-6   * eye(3), ...   % wf      penalty
             1e-6   * eye(3));      % alpha   penalty

Kpq = designLevel2ControllerILMI(Ac1o2, -Bc1o2, Cc1o2, Q2, true);

fprintf('\n  Designed Kpq:\n');
disp(Kpq);

uav.kp_q1 = Kpq(1,1);
uav.kp_q2 = Kpq(2,2);
uav.kp_q3 = Kpq(3,3);
uav.Kpq   = Kpq;

%% --- Simulation setup ---------------------------------------------------
qd = [0.8535534; 0.3535534; 0.3535534; 0.1464466 ];
qd = qd / norm(qd);   % normalize to unit quaternion

q0 = [1; 0; 0; 0];    % initial attitude: hover (identity quaternion)

%% --- Compute Level-1 equilibrium for state initialization ---------------
x1_star = computeLevel1EqPoint(uav, zeros(3,1), T);

%% --- Assemble initial state vector --------------------------------------
x2_init         = zeros(26, 1);
x2_init(7:10)   = q0;
x2_init(11:end) = x1_star;

%% --- Simulate nonlinear closed-loop dynamics ----------------------------
tspan = 0:0.001:1;   % simulation horizon  (s)
[t, p, v, q, w, wr, z, wf, alpha] = simulateLevel2(uav, tspan, x2_init, qd, T);

%% --- Plot: quaternion response ------------------------------------------
clf;

subplot(2, 1, 1);
plot(t, q, 'LineWidth', 1.5);
hold on;
plot(t, repmat(qd', length(t), 1), '--k', 'LineWidth', 1.0);
hold off;
xlabel('Time (s)');
ylabel('Quaternion q');
title('Attitude Response — Quaternion');
legend('q_0', 'q_1', 'q_2', 'q_3', ...
       'q_{d,0}', 'q_{d,1}', 'q_{d,2}', 'q_{d,3}', ...
       'Location', 'best');
grid on;

%% --- Plot: angular rate response ----------------------------------------
subplot(2, 1, 2);
plot(t, w, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Angular rate (rad/s)');
title('Angular Rate Response');
legend('\omega_x', '\omega_y', '\omega_z', 'Location', 'best');
grid on;