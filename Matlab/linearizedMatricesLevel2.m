function [Ao1o2, Bo1o2, Co1o2, Ac1o2, Bc1o2, Cc1o2, Ac1c2] = linearizedMatricesLevel2(uav, T)
% LINEARIZEDMATRICESLEVEL2  Assemble the open- and closed-loop state-space
%                           matrices for the Level-2 attitude control loop.
%
%   [Ao1o2, Bo1o2, Co1o2, Ac1o2, Bc1o2, Cc1o2, Ac1c2] = ...
%       linearizedMatricesLevel2(uav, T)
%
%   Linearizes the coupled Level-1 / Level-2 dynamics 
%   equilibrium point (wd = 0, qe = 0) and returns the state-space matrices
%   for two configurations:
%     - Open Level-1,   Open Level-2   → Ao1o2, Bo1o2, Co1o2
%     - Closed Level-1, Open Level-2   → Ac1o2, Bc1o2, Cc1o2
%   The fully closed-loop matrix Ac1c2 is also returned.
%
% STATE VECTOR  x2 (19×1) — reduced order (qe0 eliminated via unit norm)
%   x2 = [ qe_vec  (3×1)  quaternion vector part             (—)     ]
%        [ w       (3×1)  body angular rate                  (rad/s) ]
%        [ wr      (4×1)  rotor speeds                       (rad/s) ]
%        [ z       (3×1)  angular-rate integrator            (rad)   ]
%        [ wf      (3×1)  filtered angular rate               (rad/s)]
%        [ alpha   (3×1)  angular acceleration               (rad/s²)]
%
% INPUT
%   uav  - UAV parameter struct (see SETDEFAULTPARAMS)
%   T    - collective thrust command at equilibrium (N)
%
% OUTPUT
%   Ao1o2  - (19×19) open-loop system matrix   (open L1, open L2)
%   Bo1o2  - (19×6)  input matrix              (open L1, open L2)
%   Co1o2  - (12×19) output matrix             (open L1, open L2)
%   Ac1o2  - (19×19) system matrix             (closed L1, open L2)
%   Bc1o2  - (19×3)  input matrix              (closed L1, open L2)
%   Cc1o2  - (3×19)  output matrix             (closed L1, open L2)
%   Ac1c2  - (19×19) closed-loop system matrix (closed L1, closed L2)
%                      Ac1c2 = Ao1o2 − Bo1o2·K2·Co1o2
%                            = Ac1o2 − Bc1o2·Kpq·Cc1o2
%
% CONTROL ARCHITECTURE
%   Level 2:  wd   = −Kpq · qe_vec
%   Level 1:  tau  = −K1  · C1 · x1
%   Combined: K2   = [ Kpq,        0,   0,   0  ]
%                    [ Kpw·Kpq,  Kpw, Kiw, Kdw  ]
%
% EXAMPLE
%   [~,~,~,~,~,~, Ac1c2] = linearizedMatricesLevel2(uav, 0.5);
%   eig(Ac1c2)
%
% See also: LINEARIZEDMATRICESLEVEL1, COMPUTELEVEL1EQPOINT, SETDEFAULTPARAMS,
%           DERIVEATTITUDEDYNAMICS

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    %% --- Gain matrices ---------------------------------------------------
    K1  = [uav.Kpw, uav.Kiw, uav.Kdw];

    K2  = [        uav.Kpq, zeros(3,3), zeros(3,3), zeros(3,3) ;
           uav.Kpw*uav.Kpq,    uav.Kpw,    uav.Kiw,    uav.Kdw ];

    %% --- Equilibrium point (wd = 0) ------------------------------
    wd = zeros(3, 1);
    [~, ~, wr_star, ~, ~, ~, ~] = computeLevel1EqPoint(uav, wd, T);

    %% --- Level-1 open-loop system matrix Ao1  (16×16) -------------------
    Jinv = uav.J \ eye(3);

    Ao1 = zeros(16, 16);

    % w-dot  (rows 1:3)
    Ao1(1:3,   1:3)  = Jinv * (QuaternionTools.skew(uav.J*wd) - QuaternionTools.skew(wd)*uav.J);
    Ao1(1:3,   4:7)  = 2 * Jinv * uav.Gtau * diag(wr_star);

    % wr-dot  (rows 4:7)
    Ao1(4:7,   4:7)  = -uav.ar * eye(4);

    % z-dot  (rows 8:10)
    Ao1(8:10,  1:3)  = eye(3);

    % wf-dot → alpha coupling  (rows 11:13)
    Ao1(11:13, 14:16) = eye(3);

    % alpha-dot  (rows 14:16) — 2nd-order Butterworth
    Ao1(14:16,  1:3)  =  uav.wc^2 * eye(3);
    Ao1(14:16, 11:13) = -uav.wc^2 * eye(3);
    Ao1(14:16, 14:16) = -sqrt(2) * uav.wc * eye(3);

    %% --- Level-1 input / output matrices --------------------------------
    B1 = [ zeros(3,3)      ;   % w
           uav.br*uav.M_w  ;   % wr
           zeros(3,3)      ;   % z
           zeros(3,3)      ;   % wf
           zeros(3,3)      ];  % alpha

    C1 = [  eye(3),   zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;   % w
           zeros(3),  zeros(3,4),    eye(3),  zeros(3,3), zeros(3,3) ;   % z
           zeros(3),  zeros(3,4), zeros(3,3), zeros(3,3),    eye(3)  ];  % alpha

    %% --- Level-1 closed-loop matrix  Ac1 --------------------------------
    Ac1 = Ao1 - B1*K1*C1;
    a2 = [0.5*eye(3), zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3)];

    %% --- Case A: Open Level-1, Open Level-2 -----------------------------
    Ao1o2 = [ zeros(3,3),  a2  ;
              zeros(16,3), Ao1 ];

    Bo1o2 = [ zeros(3,3),       zeros(3,3)      ;
              zeros(3,3),       zeros(3,3)      ;
              zeros(4,3),       uav.br*uav.M_w  ;
                -eye(3),        zeros(3,3)      ;
              zeros(3,3),       zeros(3,3)      ;
              zeros(3,3),       zeros(3,3)      ];

    Co1o2 = [   eye(3),  zeros(3), zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;
              zeros(3),   eye(3),  zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ;
              zeros(3),  zeros(3), zeros(3,4),    eye(3),  zeros(3,3), zeros(3,3) ;
              zeros(3),  zeros(3), zeros(3,4), zeros(3,3), zeros(3,3),    eye(3)  ];

    Ac1c2 = Ao1o2 - Bo1o2*K2*Co1o2;

    %% --- Case B: Closed Level-1, Open Level-2 ---------------------------
    Ac1o2 = [ zeros(3,3),  a2  ;
              zeros(16,3), Ac1 ];

    Bc1o2 = [ zeros(3,3)          ;
              zeros(3,3)          ;
              uav.br*uav.M_w*uav.Kpw ;
                -eye(3)           ;
              zeros(3,3)          ;
              zeros(3,3)          ];

    Cc1o2 = [ eye(3), zeros(3), zeros(3,4), zeros(3,3), zeros(3,3), zeros(3,3) ];

end