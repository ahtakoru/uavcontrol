function prob = checkGeometry(uav)
% CheckGeometry  Verify UAV rotor-placement geometric conditions.
%
%   prob = check_geometric_requirements(uav)
%
%   Checks whether the rotor positions satisfy the convexity/orientation
%   conditions required for controllability of the torque-allocation matrix.
%   Condition 1 is mandatory; the Reference Condition is informational only.
%
% INPUT
%   uav  - UAV parameter struct with fields:
%            .p1, .p2, .p3, .p4  (3×1) rotor position vectors in body frame (m)
%
% OUTPUT
%   prob - Scalar flag:  0 = all mandatory conditions satisfied
%                        1 = Condition 1 violated (configuration invalid)
%
% ALGORITHM
%   Each sub-check calls orientation(A, B, C), which returns the sign of the
%   scalar triple product (or 2-D cross product) of the three position vectors.
%   A strictly positive result indicates correct half-plane placement.
%
% EXAMPLE
%   prob = checkGeometry(uav);
%   if prob, error('Invalid rotor geometry — check p1..p4.'); end
%
% See also: ORIENTATION, COMPUTELEVEL1EQPOINT

% --------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% --------------------------------------------------------------------------
    fprintf('Actuator Placement Geometry Requirements Validation.\n');
    prob = 0;

    %% --- Condition 1: mandatory geometry check ----------------------------
    cond1 = orientation(-uav.p2,  uav.p4, -uav.p1) > 0 && ...
            orientation(-uav.p1,  uav.p3, -uav.p2) > 0 && ...
            orientation(-uav.p2,  uav.p4,  uav.p3) > 0 && ...
            orientation(-uav.p1,  uav.p3,  uav.p4) > 0;

    if cond1
        fprintf('  [PASS] Condition 1 satisfied.\n');
    else
        prob = 1;
        fprintf('  [FAIL] Condition 1 NOT satisfied — rotor geometry is invalid.\n');
    end

    %% --- Reference Condition: informational only --------------------------
    cond_ref = orientation( uav.p1,  uav.p4,  uav.p2) > 0 && ...
               orientation( uav.p2,  uav.p3,  uav.p1) > 0 && ...
               orientation( uav.p1,  uav.p4,  uav.p3) > 0 && ...
               orientation( uav.p2,  uav.p3,  uav.p4) > 0;

    if cond_ref
        fprintf('  [INFO] Reference condition satisfied.\n\n');
    else
        fprintf('  [INFO] Reference condition NOT satisfied (non-critical).\n\n');
    end

end