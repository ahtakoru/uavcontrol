% EXAMPLE1  Stability analysis of the Level-1 angular-rate control loop
%           for a custom mixer matrix configuration.
%
% PURPOSE
%   This script demonstrates how to:
%     1. Load default UAV parameters and override the mixer matrix.
%     2. Validate the mixer and rotor geometry.
%     3. Linearize the Level-1 closed-loop dynamics at an equilibrium point.
%     4. Check exponential stability via closed-loop eigenvalues.
%
% See also: SETDEFAULTPARAMS, CHECKGEOMETRY, CHECKMIXER,
%           COMPUTELEVEL1EQPOINT, LINEARIZEDMATRICESLEVEL1

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

clear;  clc;

%% --- Load default parameters --------------------------------------------
uav = setDefaultParams();

%% --- Override mixer coefficients ----------------------------------------
uav.m11 = -0.1;  uav.m12 =  0.7;  uav.m13 =  1.0;  uav.m14 =  1.0;
uav.m21 =  0.7;  uav.m22 =  1.0;  uav.m23 = -0.7;  uav.m24 = -0.1;
uav.m31 = -0.1;  uav.m32 =  1.0;  uav.m33 =  1.0;  uav.m34 = -0.1;
uav.m41 =  0.7;  uav.m42 =  0.7;  uav.m43 = -0.7;  uav.m44 = -0.1;

uav.M = [ -uav.m11,  uav.m12,  uav.m13,  uav.m14 ;
           uav.m21, -uav.m22,  uav.m23,  uav.m24 ;
           uav.m31,  uav.m32, -uav.m33,  uav.m34 ;
          -uav.m41, -uav.m42, -uav.m43,  uav.m44 ];

uav.M_w = uav.M(:, 1:3);
uav.M_T = uav.M(:, 4);

%% --- Validate geometry and mixer ----------------------------------------
checkGeometry(uav);
checkMixer(uav);

%% --- Define operating point ---------------------------------------------
wd = [0; 0; 0];   % desired angular rate  (rad/s)
T  = 0.5;         % collective thrust     (N, or normalized)

%% --- Linearize Level-1 closed-loop dynamics -----------------------------
[Ao1, B1, C1, K1, Ac1] = linearizedMatricesLevel1(uav, wd, T);

%% --- Stability analysis -------------------------------------------------
eigvals     = eig(Ac1);
maxRealPart = max(real(eigvals));

fprintf('  Closed-loop eigenvalue analysis:\n');
fprintf('    max(Re(eig(Ac1))) = %.6f\n\n', maxRealPart);

if maxRealPart < 0
    fprintf('  [PASS] Equilibrium is exponentially stable.\n\n');
else
    fprintf('  [FAIL] Equilibrium is NOT exponentially stable.\n\n');
end