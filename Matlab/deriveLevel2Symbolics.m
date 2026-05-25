% DERIVE_LEVEL2_SYMBOLICS  Symbolic derivation of the Level-2 attitude
%                          control loop dynamics and linearisation.
%
% PURPOSE
%   This script derives and verifies, in closed symbolic form:
%     1. The nonlinear closed-loop dynamics  fc2(x2, T)
%     2. The reduced-order dynamics  fc2_red  (qe0 eliminated via unit norm)
%     3. The full Jacobian  Ac1c2 = d(fc2_red)/d(x2_red) |_{x2_red*}
%     4. The analytic block form under open Level-1 / open Level-2:
%           Ac1c2_block = Ao1o2 - Bo1o2*K2*Co1o2
%     5. The analytic block form under closed Level-1 / open Level-2:
%           Ac1c2_block = Ac1o2 - Bc1o2*Kpq*Cc1o2
%
% STATE VECTORS
%   Full state  x2 (20×1):
%     x2 = [ qe      (4×1)  attitude error quaternion        (—)     ]
%          [ w       (3×1)  body angular rate                (rad/s) ]
%          [ wr      (4×1)  rotor speeds                     (rad/s) ]
%          [ z       (3×1)  angular-rate integrator          (rad)   ]
%          [ wf      (3×1)  filtered angular rate             (rad/s)]
%          [ alpha   (3×1)  angular acceleration             (rad/s²)]
%
%   Reduced state  x2_red (19×1):
%     x2_red = [ qe_vec  (3×1)  quaternion imaginary part    (—)     ]
%              [ w       (3×1)  ...                                   ]
%              [ wr      (4×1)  ...                                   ]
%              [ z       (3×1)  ...                                   ]
%              [ wf      (3×1)  ...                                   ]
%              [ alpha   (3×1)  ...                                   ]
%     (qe0 eliminated via unit-norm constraint: qe0 = sqrt(1-qe1²-qe2²-qe3²))
%
% CONTROL ARCHITECTURE
%   Level 2:  wd   = -Kpq * qe_vec          (attitude → rate reference)
%   Level 1:  tau  = -K1  * C1 * x1 + ...  (rate PID with Butterworth)
%
% VERIFICATION CHECKS (run automatically, results printed to console)
%   Ac1c2 - Ac1c2_block(Ao1o2) = 0   open/open  block form
%   Ac1c2 - Ac1c2_block(Ac1o2) = 0   closed/open block form
%
% DEPENDENCIES
%   Symbolic Math Toolbox
%   QuaternionTools.skew(),  QuaternionTools.multiplication(),
%   QuaternionTools.embed(), QuaternionTools.imag()
%
% See also: DERIVE_LEVEL1_SYMBOLICS, LEVEL1_COMPACT_FORMS, COMPUTE_EQ_POINT

% --------------------------------------------------------------------------
% Copyright (c) 2024 controlUAV contributors
% SPDX-License-Identifier: MIT
% --------------------------------------------------------------------------

clear;  clc;

%% --- Symbolic variables --------------------------------------------------

% Physical states
syms wx wy wz wr1 wr2 wr3 wr4 real

% Attitude error quaternion  [scalar; vector]
syms qe0 qe1 qe2 qe3 real

% UAV physical parameters
syms Jxx Jyy Jzz Jxy Jxz Jyz real
syms ct cd ar br               real

% Rotor geometry (moment arms)
syms xl1 xl2 xl3 xl4 real
syms yl1 yl2 yl3 yl4 real

% Exogenous input
syms T real

% Dynamic controller states
syms zx zy zz             real   % integrator
syms wfx wfy wfz          real   % Butterworth filter state
syms alphax alphay alphaz real   % filtered angular acceleration

% PID gains — Level 1
syms kp_wx kp_wy kp_wz real
syms ki_wx ki_wy ki_wz real
syms kd_wx kd_wy kd_wz real

% Proportional gains — Level 2 (attitude)
syms kp_q1 kp_q2 kp_q3 real

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

%% --- Aerodynamic torque and thrust matrix --------------------------------
G = [ -ct*yl1,  ct*yl2,  ct*yl3, -ct*yl4 ;
       ct*xl1, -ct*xl2,  ct*xl3, -ct*xl4 ;
       cd,      cd,     -cd,     -cd      ;
       ct,      ct,      ct,      ct      ];

Gtau = G(1:3, :);   % torque rows
Gf   = G(4,   :);   %#ok<NASGU>  thrust row — unused here, defined for reference

%% --- Inertia tensor ------------------------------------------------------
J = [ Jxx, Jxy, Jxz ;
      Jxy, Jyy, Jyz ;
      Jxz, Jyz, Jzz ];

%% --- Rotor position vectors (body x-y plane, for reference) -------------
p1 = [ xl1;  yl1];   %#ok<NASGU>
p2 = [-xl2; -yl2];   %#ok<NASGU>
p3 = [ xl3; -yl3];   %#ok<NASGU>
p4 = [-xl4;  yl4];   %#ok<NASGU>

%% --- State vectors and gain matrices ------------------------------------
qe    = [qe0; qe1; qe2; qe3];
w     = [wx;  wy;  wz];
wr    = [wr1; wr2; wr3; wr4];
z     = [zx;  zy;  zz];
wf    = [wfx; wfy; wfz];
alpha = [alphax; alphay; alphaz];

x2     = [qe; w; wr; z; wf; alpha];          % full    20×1 state vector
x2_red = [QuaternionTools.imag(qe); w; wr; z; wf; alpha];  % reduced 19×1

% Level-1 PID gain matrix  K1 (3×9)
Kpw = diag([kp_wx, kp_wy, kp_wz]);
Kiw = diag([ki_wx, ki_wy, ki_wz]);
Kdw = diag([kd_wx, kd_wy, kd_wz]);
K1  = [Kpw, Kiw, Kdw];

% Level-2 attitude gain matrix  Kpq (3×3)
Kpq = diag([kp_q1, kp_q2, kp_q3]);

% Combined gain matrix  K2 (6×12) for the open/open block form
K2 = [      Kpq, zeros(3,3), zeros(3,3), zeros(3,3) ;
        Kpw*Kpq,        Kpw,        Kiw,        Kdw  ];

%% --- Mixer matrix and decomposition --------------------------------------
M = [ -m11,  m12,  m13, m14 ;
       m21, -m22,  m23, m24 ;
       m31,  m32, -m33, m34 ;
      -m41, -m42, -m43, m44 ];

M_w = M(:, 1:3);
M_T = M(:, 4);     %#ok<NASGU>  thrust column — unused here

%% --- Level-2 control law ------------------------------------------------
wd = -Kpq * qe(2:4);   % attitude error → angular-rate reference

%% --- Level-1 control law ------------------------------------------------
taudx = -kp_wx*(wx - wd(1)) - ki_wx*zx - kd_wx*alphax;
taudy = -kp_wy*(wy - wd(2)) - ki_wy*zy - kd_wy*alphay;
taudz = -kp_wz*(wz - wd(3)) - ki_wz*zz - kd_wz*alphaz;

PWM = M * [taudx; taudy; taudz; T];

%% --- Nonlinear dynamics  fc2(x2, T) -------------------------------------
dqe   = (sym(1)/2) * QuaternionTools.multiplication(qe, QuaternionTools.embed(w));
dw    = inv(J) * (cross(-w, J*w) + Gtau*wr.^2);
dwr   = -ar*wr + br*PWM;
dz    = w - wd;
dwf   = alpha;
dalpha = -wc^2*wf - sqrt(sym(2))*wc*alpha + wc^2*w;   % 2nd-order Butterworth

fc2 = [dqe; dw; dwr; dz; dwf; dalpha];

%% --- Reduced-order dynamics  (eliminate qe0 via unit-norm constraint) ---
%   qe0 = sqrt(1 - qe1² - qe2² - qe3²)
fc2_red = subs(fc2(2:end), qe0, sqrt(1 - qe1^2 - qe2^2 - qe3^2));

%% --- Equilibrium point  x2_red*  (hover: qe_vec = 0, w = 0) ------------
qe_star    = sym(zeros(3, 1));
w_star     = sym(zeros(3, 1));
wr_star    = [wr1_star; wr2_star; wr3_star; wr4_star];
z_star     = [zx_star;  zy_star;  zz_star];
wf_star    = sym(zeros(3, 1));
alpha_star = sym(zeros(3, 1));

x2_red_star = [qe_star; w_star; wr_star; z_star; wf_star; alpha_star];

%% --- Jacobian linearisation ---------------------------------------------
fprintf('Computing Jacobian of fc2_red (this may take a moment) ...\n');
Jfc2_red = simplify(jacobian(fc2_red, x2_red));
Ac1c2    = subs(Jfc2_red, x2_red, x2_red_star);

%% --- Level-1 open-loop block matrices (at hover: wd1 = 0) ---------------
wd1 = sym(zeros(3, 1));

Ao1 = sym(zeros(16, 16));

% w-dot  (rows 1:3)
Ao1(1:3,  1:3)  = inv(J) * (QuaternionTools.skew(J*wd1) - QuaternionTools.skew(wd1)*J);
Ao1(1:3,  4:7)  = 2 * (inv(J) * Gtau) * diag(wr_star);

% wr-dot  (rows 4:7)
Ao1(4:7,  4:7)  = -ar * eye(4);

% z-dot  (rows 8:10)
Ao1(8:10, 1:3)  = eye(3);

% wf-dot → alpha coupling  (rows 11:13)
Ao1(11:13, 14:16) = eye(3);

% alpha-dot  (rows 14:16)  — 2nd-order Butterworth
Ao1(14:16,  1:3)  =  wc^2 * eye(3);
Ao1(14:16, 11:13) = -wc^2 * eye(3);
Ao1(14:16, 14:16) = -sqrt(sym(2)) * wc * eye(3);

B1 = [ zeros(3,3) ; br*M_w ; zeros(3,3) ; zeros(3,3) ; zeros(3,3) ];

C1 = [  eye(3),   zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;
       zeros(3),  zeros(3,4),    eye(3),  zeros(3,3), zeros(3,3) ;
       zeros(3),  zeros(3,4), zeros(3,3), zeros(3,3),    eye(3)  ];

Ac1 = Ao1 - B1*K1*C1;   % Level-1 closed-loop matrix

%% --- Case A: Open Level-1, Open Level-2 ----------------------------------
%
%   Full 19×19 system matrix  Ao1o2
%   Partitioned as:  [ qe_vec dynamics  |  coupling to x1 ]
%                    [ 0               |  Ao1             ]

% Quaternion kinematics row (linearised):  dqe_vec ≈ (1/2)*I*w  at hover
a212 = sym([1/2*eye(3), zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3)]);

Ao1o2 = [ zeros(3,3), a212 ;
          zeros(16,3), Ao1  ];

% Input matrix for [wd; tau] entering Level-1 open-loop
Bo1o2 = [ zeros(3,3),  zeros(3,3) ;
          zeros(3,3),  zeros(3,3) ;
          zeros(4,3),  br*M_w     ;
            -eye(3),   zeros(3,3) ;
          zeros(3,3),  zeros(3,3) ;
          zeros(3,3),  zeros(3,3) ];

% Output matrix: selects [qe_vec; w; z; alpha] from [qe_vec; x1]
Co1o2 = [   eye(3),  zeros(3),  zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;
           zeros(3),   eye(3),  zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;
           zeros(3),  zeros(3), zeros(3,4),    eye(3),  zeros(3,3), zeros(3,3) ;
           zeros(3),  zeros(3), zeros(3,4), zeros(3,3), zeros(3,3),    eye(3)  ];

Ac1c2_OO = Ao1o2 - Bo1o2*K2*Co1o2;

%% --- Verification 1: open/open block form --------------------------------
fprintf('Checking Ac1c2 == Ao1o2 - Bo1o2*K2*Co1o2  (open L1, open L2) ...\n');
residual_OO = simplify(Ac1c2 - Ac1c2_OO);
if all(residual_OO == sym(0), 'all')
    fprintf('  [PASS] Ac1c2 - (Ao1o2 - Bo1o2*K2*Co1o2) = 0\n\n');
else
    fprintf('  [FAIL] Nonzero residual detected.\n\n');
    disp(residual_OO);
end

%% --- Case B: Closed Level-1, Open Level-2 --------------------------------
%
%   Full 19×19 system matrix  Ac1o2
%   Partitioned as:  [ qe_vec dynamics  |  coupling to x1 ]
%                    [ 0               |  Ac1             ]

Ac1o2 = [ zeros(3,3), a212 ;
          zeros(16,3), Ac1  ];

% Input matrix: wd enters via the rate-error and Kpw feedforward
Bc1o2 = [ zeros(3,3)  ;
          zeros(3,3)  ;
          br*M_w*Kpw  ;
            -eye(3)   ;
          zeros(3,3)  ;
          zeros(3,3)  ];

% Output matrix: selects w from [qe_vec; x1]
Cc1o2 = [ eye(3), zeros(3), zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ];

Ac1c2_CO = Ac1o2 - Bc1o2*Kpq*Cc1o2;

%% --- Verification 2: closed/open block form ------------------------------
fprintf('Checking Ac1c2 == Ac1o2 - Bc1o2*Kpq*Cc1o2  (closed L1, open L2) ...\n');
residual_CO = simplify(Ac1c2 - Ac1c2_CO);
if all(residual_CO == sym(0), 'all')
    fprintf('  [PASS] Ac1c2 - (Ac1o2 - Bc1o2*Kpq*Cc1o2) = 0\n\n');
else
    fprintf('  [FAIL] Nonzero residual detected.\n\n');
    disp(residual_CO);
end

fprintf('Symbolic derivation complete.\n');