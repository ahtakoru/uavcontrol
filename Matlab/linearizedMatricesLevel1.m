function [Ao1, B1, C1, K1, Ac1] = linearizedMatricesLevel1(uav, wd, T)
% LEVEL1_COMPACT_FORMS  Assemble the open- and closed-loop state-space matrices
%                       for the Level-1 angular-rate control loop.
%
%   [Ao1, B1, C1, K1, Ac1] = level1_compact_forms(uav, wd, T)
%
%   Linearises the Level-1 dynamics around the equilibrium point computed by
%   COMPUTE_EQ_POINT and returns the compact state-space matrices used for
%   LQR / eigenvalue-based controller design.
%
% STATE VECTOR  x1 (16×1)
%   x1 = [ w        (3×1)  body angular rate          (rad/s)   ]
%        [ wr       (4×1)  rotor speeds               (rad/s)   ]
%        [ z        (3×1)  angular-rate integrator    (rad)     ]
%        [ wf       (3×1)  filtered angular rate       (rad/s)  ]
%        [ alpha    (3×1)  angular acceleration        (rad/s²) ]
%
% INPUT
%   uav  - UAV parameter struct with fields:
%            .J              (3×3) inertia matrix
%            .Gtau           (3×4) torque-allocation matrix
%            .M_w            (4×4) rotor-speed mixing matrix
%            .M_T            (4×1) thrust-coefficient vector
%            .ar, .br        rotor first-order model coefficients (scalars)
%            .kp_wx/wy/wz    proportional PID gains (scalars)
%            .ki_wx/wy/wz    integral     PID gains (scalars)
%            .kd_wx/wy/wz    derivative   PID gains (scalars)
%            .wc             Butterworth filter cut-off frequency (rad/s)
%   wd   - (3×1) desired angular velocity at equilibrium (rad/s)
%   T    - collective thrust command at equilibrium (N)
%
% OUTPUT
%   Ao1  - (16×16) open-loop system matrix  (linearised at equilibrium)
%   B1   - (16×3)  input matrix  (rotor-speed commands → states)
%   C1   - (9×16)  output matrix  [w; z; alpha]
%   K1   - (3×9)   PID gain matrix  [Kp, Ki, Kd]
%   Ac1  - (16×16) closed-loop system matrix  Ao1 − B1·K1·C1
%
% EQUATIONS
%   Open-loop:    dx1/dt = Ao1·x1 + B1·u
%   Output:       y      = C1·x1
%   Control law:  u      = −K1·y
%   Closed-loop:  dx1/dt = Ac1·x1,  Ac1 = Ao1 − B1·K1·C1
%
% EXAMPLE
%   [Ao1, B1, C1, K1, Ac1] = level1_compact_forms(uav, [0;0;1], 9.81);
%   eig(Ac1)   % check closed-loop eigenvalues
%
% See also: COMPUTE_EQ_POINT, SKEW, CHECK_MIXER_REQUIREMENTS

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    %% --- PID gain matrices -----------------------------------------------
    Kpw = diag([uav.kp_wx, uav.kp_wy, uav.kp_wz]);
    Kiw = diag([uav.ki_wx, uav.ki_wy, uav.ki_wz]);
    Kdw = diag([uav.kd_wx, uav.kd_wy, uav.kd_wz]);

    K1 = [Kpw, Kiw, Kdw];   % (3×9) full PID gain matrix

    %% --- Input matrix B1  (16×3) -----------------------------------------
    % Only the rotor-speed rows (4:7) are driven by the control input.
    B1 = [ zeros(3, 3)         ;   % w      rows
           uav.br * uav.M_w    ;   % wr     rows
           zeros(3, 3)         ;   % z      rows
           zeros(3, 3)         ;   % wf     rows
           zeros(3, 3)         ];  % alpha  rows

    %% --- Output matrix C1  (9×16) ----------------------------------------
    % Outputs: angular rate w (rows 1:3), integrator z (rows 8:10),
    %          angular acceleration alpha (rows 14:16).
    C1 = [ eye(3),    zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;   % w
           zeros(3),  zeros(3,4),    eye(3),  zeros(3,3), zeros(3,3) ;   % z
           zeros(3),  zeros(3,4), zeros(3,3), zeros(3,3),    eye(3)  ];  % alpha

    %% --- Equilibrium point -----------------------------------------------
    [~, ~, wr_star, ~, ~, ~, ~] = computeLevel1EqPoint(uav, wd, T);

    %% --- Open-loop system matrix Ao1  (16×16) ----------------------------
    Jinv = uav.J \ eye(3);   % compute once; avoids repeated inv() calls

    Ao1 = zeros(16, 16);

    % w-dot  rows 1:3
    Ao1(1:3,  1:3)  = Jinv * (QuaternionTools.skew(uav.J * wd) - QuaternionTools.skew(wd) * uav.J);
    Ao1(1:3,  4:7)  = 2 * Jinv * uav.Gtau * diag(wr_star);

    % wr-dot  rows 4:7
    Ao1(4:7,  4:7)  = -uav.ar * eye(4);

    % z-dot  rows 8:10
    Ao1(8:10, 1:3)  = eye(3);

    % wf-dot  rows 11:13  — identity coupling to alpha (handled by alpha rows)

    % alpha-dot  rows 14:16  (2nd-order Butterworth state)
    Ao1(11:13, 14:16) = eye(3);
    Ao1(14:16,  1:3)  =  uav.wc^2 * eye(3);
    Ao1(14:16, 11:13) = -uav.wc^2 * eye(3);
    Ao1(14:16, 14:16) = -sqrt(2) * uav.wc * eye(3);

    %% --- Closed-loop system matrix ---------------------------------------
    Ac1 = Ao1 - B1 * K1 * C1;

end