% DERIVE_LEVEL1_SYMBOLICS  Symbolic derivation of the Level-1 angular-rate
%                          control loop dynamics, compact form, and linearization.
%
% PURPOSE
%   This script derives and verifies, in closed symbolic form:
%     1. The nonlinear equations of motion  f(q, u)
%     2. The compact affine form  f = fo1 - B1*K1*C1*q + B1T*T + B1w*wd
%     3. The open-loop Jacobian  Ao1 = df_o1/dq |_{q*}
%     4. The closed-loop Jacobian  Ac1 = df/dq   |_{q*}
%     5. The analytic block-matrix form  myAo1  (matches Ao1 entry-by-entry)
%
% STATE VECTOR  q (16×1)
%   q = [ w        (3×1)  body angular rate           (rad/s)  ]
%       [ wr       (4×1)  rotor speeds                (rad/s)  ]
%       [ z        (3×1)  angular-rate integrator     (rad)    ]
%       [ wf       (3×1)  filtered angular rate        (rad/s) ]
%       [ alpha    (3×1)  angular acceleration         (rad/s²)]
%
% VERIFICATION CHECKS (uncomment to run)
%   fc1 - f_compact = 0          compact form correctness
%   Ac1 - (Ao1 - B1*K1*C1) = 0  closed-loop Jacobian identity
%   Ao1 - Ao1_block = 0          analytic block form correctness
%
% DEPENDENCIES
%   Symbolic Math Toolbox
%   QuaternionTools.skew()  (local utility — see toolbox)
%
% See also: LEVEL1_COMPACT_FORMS, COMPUTE_EQ_POINT, ILMI_SOFB

% --------------------------------------------------------------------------
% Copyright (c) 2024 controlUAV contributors
% SPDX-License-Identifier: MIT
% --------------------------------------------------------------------------

clear;  clc;

%% --- Symbolic variables --------------------------------------------------

% Physical states
syms wx wy wz wr1 wr2 wr3 wr4 real

% UAV physical parameters
syms Jxx Jyy Jzz Jxy Jxz Jyz real
syms ct cd ar br                   real

% Rotor geometry (moment arms)
syms xl1 xl2 xl3 xl4 real
syms yl1 yl2 yl3 yl4 real

% Reference / exogenous inputs
syms T wdx wdy wdz real

% Dynamic controller states
syms zx zy zz             real   % integrator
syms wfx wfy wfz          real   % Butterworth filter state
syms alphax alphay alphaz real   % filtered angular acceleration

% PID gains
syms kp_wx kp_wy kp_wz real
syms ki_wx ki_wy ki_wz real
syms kd_wx kd_wy kd_wz real

% Butterworth cut-off frequency
syms wc real

% Mixer coefficients (all non-negative by convention)
syms m11 m12 m13 m14 real
syms m21 m22 m23 m24 real
syms m31 m32 m33 m34 real
syms m41 m42 m43 m44 real

% Equilibrium rotor speeds and integrator states
syms wr1_star wr2_star wr3_star wr4_star real
syms zx_star  zy_star  zz_star           real

%% --- Aerodynamic force and torque generation matrix ----------------------
%
%   G maps squared rotor speeds to [tau_x; tau_y; tau_z; F_z]
%
G = [ -ct*yl1,  ct*yl2,  ct*yl3, -ct*yl4 ;
       ct*xl1, -ct*xl2,  ct*xl3, -ct*xl4 ;
       cd,      cd,     -cd,     -cd      ;
       ct,      ct,      ct,      ct      ];

Gtau = G(1:3, :);   % torque rows
Gf   = G(4,   :);   % thrust row  (kept for reference)

%% --- Inertia tensor ------------------------------------------------------
J = [ Jxx, Jxy, Jxz ;
      Jxy, Jyy, Jyz ;
      Jxz, Jyz, Jzz ];

%% --- Rotor position vectors (body x-y plane) ----------------------------
p1 = [ xl1;  yl1];
p2 = [-xl2; -yl2];
p3 = [ xl3; -yl3];
p4 = [-xl4;  yl4];

%% --- Compact state, input, and gain definitions --------------------------
w     = [wx;  wy;  wz];
wr    = [wr1; wr2; wr3; wr4];
z     = [zx;  zy;  zz];
wf    = [wfx; wfy; wfz];
alpha = [alphax; alphay; alphaz];
wd    = [wdx; wdy; wdz];

x1 = [w; wr; z; wf; alpha];   % full 16×1 state vector

Kpw = diag([kp_wx, kp_wy, kp_wz]);
Kiw = diag([ki_wx, ki_wy, ki_wz]);
Kdw = diag([kd_wx, kd_wy, kd_wz]);
K1  = [Kpw, Kiw, Kdw];       % (3×9) PID gain matrix

%% --- Mixer matrix and decomposition --------------------------------------
M = [ -m11,  m12,  m13, m14 ;
       m21, -m22,  m23, m24 ;
       m31,  m32, -m33, m34 ;
      -m41, -m42, -m43, m44 ];

M_w = M(:, 1:3);   % torque columns
M_T = M(:, 4);     % thrust column

%% --- Control law  (u = -K1 * C1 * q + feedforward) ----------------------
taudx = -kp_wx*(wx - wdx) - ki_wx*zx - kd_wx*alphax;
taudy = -kp_wy*(wy - wdy) - ki_wy*zy - kd_wy*alphay;
taudz = -kp_wz*(wz - wdz) - ki_wz*zz - kd_wz*alphaz;

PWM = M * [taudx; taudy; taudz; T];

%% --- Nonlinear dynamics  f_{c1}(x1, wd, T) -------------------------------
dw    = inv(J) * (cross(-w, J*w) + Gtau*wr.^2);
dwr   = -ar*wr + br*PWM;
dz    = w - wd;
dwf   = alpha;
dalpha = -wc^2*wf - sqrt(sym(2))*wc*alpha + wc^2*w;   % 2nd-order Butterworth

fc1 = [dw; dwr; dz; dwf; dalpha];

%% --- Compact form --------------------------------------------------------
%
%   fc1 = fo1 - B1*K1*C1*q + B1T*T + B1w*wd
%
%   fo1  : uncontrolled (open-loop) drift
%   B1   : input matrix for rotor-speed commands
%   B1w  : feedforward matrix for reference angular rate
%   B1T  : feedforward vector for thrust T

fo1 = [ inv(J) * (cross(-w, J*w) + Gtau*wr.^2) ;   % dw    (open-loop)
       -ar*wr                              ;   % dwr   (open-loop)
        w                                  ;   % dz    (reference subtracted below)
        alpha                              ;   % dwf
       -wc^2*wf - sqrt(sym(2))*wc*alpha + wc^2*w ];  % dalpha

B1  = [ zeros(3,3)  ; br*M_w ; zeros(3,3) ; zeros(3,3) ; zeros(3,3) ];
B1w = [ zeros(3,3)  ; br*M_w*Kpw ; -eye(3) ; zeros(3,3) ; zeros(3,3) ];
B1T = [ zeros(3,1)  ; br*M_T     ; zeros(3,1) ; zeros(3,1) ; zeros(3,1) ];

C1 = [  eye(3),   zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;   % w
       zeros(3),  zeros(3,4),    eye(3),  zeros(3,3), zeros(3,3) ;   % z
       zeros(3),  zeros(3,4), zeros(3,3), zeros(3,3),    eye(3)  ];  % alpha

f_compact = fo1 - B1*K1*C1*x1 + B1T*T + B1w*wd;

%% --- Verification 1: compact form correctness ---------------------------
fprintf('Checking compact form \n');
residual_compact = simplify(fc1 - f_compact);
if all(residual_compact == sym(0), 'all')
    fprintf('  [PASS] fc1 - fo1 + B1*K1*C1*x1 - B1T*T - B1w*wd = 0\n\n');
else
    fprintf('  [FAIL] Nonzero residual detected — check derivation.\n\n');
    disp(residual_compact);
end

%% --- Equilibrium point  q* ----------------------------------------------
w_star     = wd;
wr_star    = [wr1_star; wr2_star; wr3_star; wr4_star];
z_star     = [zx_star;  zy_star;  zz_star];
wf_star    = wd;
alpha_star = sym(zeros(3,1));

q_star = [w_star; wr_star; z_star; wf_star; alpha_star];

%% --- Linearization ------------------------------------------------------
fprintf('Computing Jacobians (this may take a moment) ...\n');

% Open-loop Jacobian  Ao1 = d(fo1)/dq |_{q*}
JAo1 = simplify(jacobian(fo1, x1));
Ao1  = subs(JAo1, x1, q_star);

% Closed-loop Jacobian  Ac1 = df/dq |_{q*}
JAc1 = simplify(jacobian(fc1, x1));
Ac1  = subs(JAc1, x1, q_star);

%% --- Verification 2: Ac1 = Ao1 - B1*K1*C1 ------------------------------
fprintf('Checking  Ac1 == Ao1 - B1*K1*C1 ...\n');
residual_cl = simplify(Ac1 - (Ao1 - B1*K1*C1));
if all(residual_cl == sym(0), 'all')
    fprintf('  [PASS] Ac1 - (Ao1 - B1*K1*C1) = 0\n\n');
else
    fprintf('  [FAIL] Nonzero residual detected.\n\n');
    disp(residual_cl);
end

%% --- Analytic block-matrix form of Ao1 ----------------------------------
%
%   Built entry-by-entry for use in LEVEL1_COMPACT_FORMS (numeric version).

Ao1_block = sym(zeros(16, 16));

% w-dot  (rows 1:3)
Ao1_block(1:3,  1:3)  = J \ (QuaternionTools.skew(J*wd) - QuaternionTools.skew(wd)*J);
Ao1_block(1:3,  4:7)  = 2 * (J \ Gtau) * diag(wr_star);

% wr-dot  (rows 4:7)
Ao1_block(4:7,  4:7)  = -ar * eye(4);

% z-dot  (rows 8:10)
Ao1_block(8:10, 1:3)  = eye(3);

% wf-dot → alpha coupling  (rows 11:13)
Ao1_block(11:13, 14:16) = eye(3);

% alpha-dot  (rows 14:16)  — 2nd-order Butterworth
Ao1_block(14:16,  1:3)  =  wc^2 * eye(3);
Ao1_block(14:16, 11:13) = -wc^2 * eye(3);
Ao1_block(14:16, 14:16) = -sqrt(sym(2)) * wc * eye(3);

%% --- Verification 3: analytic block form matches Jacobian ---------------
fprintf('Checking  Ao1 equals to block matrix representation ...\n');
residual_Ao1 = simplify(Ao1 - Ao1_block);
if all(residual_Ao1 == sym(0), 'all')
    fprintf('  [PASS] Ao1 - Ao1_block = 0\n\n');
else
    fprintf('  [FAIL] Nonzero residual detected.\n\n');
    disp(residual_Ao1);
end

fprintf('Symbolic derivation complete.\n');