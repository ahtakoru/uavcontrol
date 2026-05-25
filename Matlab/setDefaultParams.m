function uav = setDefaultParams()
% SET_PARAMETERS  Define and assemble all physical and control parameters
%                 for the controlUAV quadrotor platform.
%
%   uav = set_default_params()
%
%   Returns a struct containing the UAV physical parameters, rotor geometry,
%   aerodynamic coefficients, actuator model, mixer matrix, PID gains, and
%   all derived composite matrices used throughout the controlUAV toolbox.
%
%   Default mechanical parameters correspond to the X500 quadrotor model
%   from the PX4-Gazebo simulation environment. Default PID gains follow
%   the PX4 controller parameter conventions.
%
% OUTPUT
%   uav  - Parameter struct with fields grouped as follows:
%
%   INERTIA
%     .Jxx, .Jyy, .Jzz          principal moments of inertia  (kg·m²)
%     .Jxy, .Jxz, .Jyz          products of inertia           (kg·m²)
%     .J                         (3×3) full inertia tensor
%     .m                         total mass                    (kg)
%
%   AERODYNAMICS
%     .ct                        thrust coefficient            (N·s²/rad²)
%     .cd                        drag-torque coefficient       (N·m·s²/rad²)
%
%   ACTUATOR MODEL  (first-order rotor dynamics: dwr = -ar*wr + br*PWM)
%     .ar                        rotor pole                    (rad/s)
%     .br                        rotor input gain              (rad/s per unit)
%
%   ROTOR GEOMETRY  (moment arms in body x-y plane)
%     .xl1–.xl4                  x-axis moment arms            (m)
%     .yl1–.yl4                  y-axis moment arms            (m)
%     .p1–.p4                    (2×1) rotor position vectors  (m)
%
%   AERODYNAMIC MATRICES
%     .G                         (4×4) full force/torque matrix
%     .Gtau                      (3×4) torque allocation matrix
%     .Gf                        (1×4) thrust row
%
%   MIXER MATRIX
%     .m11–.m44                  mixer coefficients (non-negative scalars)
%     .M                         (4×4) mixer matrix
%     .M_w                       (4×3) torque columns of M
%     .M_T                       (4×1) thrust column of M
%
%   PID GAINS — LEVEL 1 (angular rate)
%     .kp_wx/wy/wz               proportional gains
%     .ki_wx/wy/wz               integral gains
%     .kd_wx/wy/wz               derivative gains
%     .Kpw, .Kiw, .Kdw           (3×3) diagonal gain matrices
%
%   PID GAINS — LEVEL 2 (attitude)
%     .kp_q1/q2/q3               proportional gains
%     .Kpq                       (3×3) diagonal gain matrix
%
%   BUTTERWORTH FILTER
%     .wc                        cut-off frequency             (rad/s)
%
%   RATE LIMITS
%     .wdmaxx/y/z                maximum angular-rate reference (rad/s)
%     .wdmax                     (3×1) collected rate limit vector
%
%   THRUST LIMITS
%     .Tmin, .Tmax               collective thrust bounds       (N or normalised)
%
% EXAMPLE
%   uav = set_parameters();
%   prob = check_mixer_requirements(uav);
%
% See also: CHECK_MIXER_REQUIREMENTS, CHECK_GEOMETRIC_REQUIREMENTS,
%           COMPUTE_EQ_POINT, LEVEL1_COMPACT_FORMS

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

%% --- Inertia -------------------------------------------------------------
uav.Jxx =  21.7e-3;   % (kg·m²)
uav.Jyy =  21.7e-3;
uav.Jzz =  40.0e-3;
uav.Jxy =  0;
uav.Jxz =  0;
uav.Jyz =  0;

uav.m   =  2.0643;    % total mass (kg)

%% --- Aerodynamic coefficients -------------------------------------------
uav.ct  =  8.55e-6;   % thrust coefficient        (N·s²/rad²)
uav.cd  =  1.37e-6;   % drag-torque coefficient   (N·m·s²/rad²)

%% --- Actuator model  (first-order: dwr = -ar*wr + br*PWM) ---------------
uav.ar  =  80;
uav.br  =  1000 * uav.ar;

%% --- Rotor geometry (moment arms) ----------------------------------------
uav.xl1 = 0.174;   uav.xl2 = 0.174;   uav.xl3 = 0.174;   uav.xl4 = 0.174;
uav.yl1 = 0.174;   uav.yl2 = 0.174;   uav.yl3 = 0.174;   uav.yl4 = 0.174;

uav.p1 = [ uav.xl1;  uav.yl1];
uav.p2 = [-uav.xl2; -uav.yl2];
uav.p3 = [ uav.xl3; -uav.yl3];
uav.p4 = [-uav.xl4;  uav.yl4];

%% --- Mixer coefficients --------------------------------------------------
uav.m11 = 0.7071;   uav.m12 = 0.7071;   uav.m13 = 1;   uav.m14 = 1;
uav.m21 = 0.7071;   uav.m22 = 0.7071;   uav.m23 = 1;   uav.m24 = 1;
uav.m31 = 0.7071;   uav.m32 = 0.7071;   uav.m33 = 1;   uav.m34 = 1;
uav.m41 = 0.7071;   uav.m42 = 0.7071;   uav.m43 = 1;   uav.m44 = 1;

%% --- PID gains — Level 1 (angular rate) ---------------------------------
uav.kp_wx = 0.15;   uav.ki_wx = 0.20;   uav.kd_wx = 0.003;
uav.kp_wy = 0.15;   uav.ki_wy = 0.20;   uav.kd_wy = 0.003;
uav.kp_wz = 0.20;   uav.ki_wz = 0.10;   uav.kd_wz = 0.000;

%% --- PID gains — Level 2 (attitude) -------------------------------------
uav.kp_q1 = 4.0;
uav.kp_q2 = 4.0;
uav.kp_q3 = 2.8;

%% --- Butterworth filter cut-off frequency --------------------------------
uav.wc = 2*pi*20;    % 20 Hz  (rad/s)

%% --- Angular rate reference limits --------------------------------------
uav.wdmaxx = 220 * pi/180;   % (rad/s)
uav.wdmaxy = 220 * pi/180;
uav.wdmaxz = 200 * pi/180;
uav.wdmax  = [uav.wdmaxx; uav.wdmaxy; uav.wdmaxz];

%% --- Thrust limits -------------------------------------------------------
uav.Tmin = 0;
uav.Tmax = 1;

%% --- Derived composite matrices -----------------------------------------

% Inertia tensor
uav.J = [ uav.Jxx, uav.Jxy, uav.Jxz ;
          uav.Jxy, uav.Jyy, uav.Jyz ;
          uav.Jxz, uav.Jyz, uav.Jzz ];

% Aerodynamic torque and thrust matrix  G (4×4)
uav.G = [ -uav.ct*uav.yl1,  uav.ct*uav.yl2,  uav.ct*uav.yl3, -uav.ct*uav.yl4 ;
           uav.ct*uav.xl1, -uav.ct*uav.xl2,  uav.ct*uav.xl3, -uav.ct*uav.xl4 ;
                   uav.cd,          uav.cd,         -uav.cd,         -uav.cd  ;
                   uav.ct,          uav.ct,          uav.ct,          uav.ct  ];

uav.Gtau = uav.G(1:3, :);   % torque allocation matrix  (3×4)
uav.Gf   = uav.G(4,   :);   % thrust row                (1×4)

% Mixer matrix  M (4×4)
uav.M = [ -uav.m11,  uav.m12,  uav.m13,  uav.m14 ;
           uav.m21, -uav.m22,  uav.m23,  uav.m24 ;
           uav.m31,  uav.m32, -uav.m33,  uav.m34 ;
          -uav.m41, -uav.m42, -uav.m43,  uav.m44 ];

uav.M_w = uav.M(:, 1:3);   % torque columns  (4×3)
uav.M_T = uav.M(:, 4);     % thrust column   (4×1)

% PID gain matrices
uav.Kpw = diag([uav.kp_wx, uav.kp_wy, uav.kp_wz]);
uav.Kiw = diag([uav.ki_wx, uav.ki_wy, uav.ki_wz]);
uav.Kdw = diag([uav.kd_wx, uav.kd_wy, uav.kd_wz]);
uav.Kpq = diag([uav.kp_q1, uav.kp_q2, uav.kp_q3]);

end