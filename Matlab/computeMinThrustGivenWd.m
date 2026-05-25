function Tmin = computeMinThrustGivenWd(uav, wd)
% COMPUTEMINTHRUST GIVENWD  Compute the minimum allowable collective thrust
%                           command for a given angular rate reference wd.
%
%   Tmin = computeMinThrustGivenWd(uav, wd)
%
%   Computes the minimum collective thrust T such that the rotor speed
%   equilibrium point remains physically admissible (all rotor speeds
%   non-negative) for the specified angular rate reference wd.
%
% INPUT
%   uav  - UAV parameter struct (see SETDEFAULTPARAMS) with fields:
%            .Gtau          (3×4) torque allocation matrix
%            .J             (3×3) inertia tensor
%            .p1, .p2, .p3, .p4  (2×1) rotor position vectors  (m)
%            .ar, .br       rotor first-order model coefficients
%            .m14, .m24, .m34, .m44  thrust mixer coefficients
%   wd   - (3×1) desired angular rate reference  (rad/s)
%
% OUTPUT
%   Tmin - scalar minimum allowable collective thrust command  (N)
%
% ALGORITHM
%   1. Compute the minimum-norm torque allocation:
%        Ginv = Gtau' * (Gtau * Gtau')^{-1}   (right pseudoinverse)
%   2. Project the gyroscopic torque cross(wd, J*wd) onto rotor space:
%        vp = Ginv * cross(wd, J*wd)
%   3. Compute the orientation coefficients vn from rotor geometry.
%   4. Find the minimum shift cmin such that vp + cmin*vn >= 0.
%   5. Convert the shifted rotor speed vector to a minimum thrust value.
%
% EXAMPLE
%   uav  = setDefaultParams();
%   wd   = [0.5; 0.3; 0.1];
%   Tmin = computeMinThrustGivenWd(uav, wd);
%   fprintf('Minimum thrust: %.4f\n', Tmin);
%
% See also: COMPUTEMINTHRUSTWORSTCASE, COMPUTELEVEL1EQPOINT, CHECKGEOMETRY

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    %% --- Right pseudoinverse of torque allocation matrix ----------------
    Ginv = uav.Gtau'*inv(uav.Gtau*uav.Gtau');

    %% --- Orientation coefficients from rotor geometry -------------------
    vn = [ orientation(-uav.p2,  uav.p4,  uav.p3) ;
           orientation(-uav.p1,  uav.p3,  uav.p4) ;
           orientation(-uav.p2,  uav.p4, -uav.p1) ;
           orientation(-uav.p1,  uav.p3, -uav.p2) ];

    %% --- Gyroscopic torque projected onto rotor speed space -------------
    vp = Ginv * cross(wd, uav.J * wd);

    %% --- Minimum shift to ensure non-negative rotor speeds --------------
    cmin = max(-vp ./ vn);
    s    = vp + cmin * vn;
    s    = s - min(s) * ones(4, 1);   % shift so minimum entry is zero

    %% --- Convert to minimum thrust command ------------------------------
    Tmin = uav.ar * sum(sqrt(s));
    Tmin = Tmin / (uav.br * (uav.m14 + uav.m24 + uav.m34 + uav.m44));

end