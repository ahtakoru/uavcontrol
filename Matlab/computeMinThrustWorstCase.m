function Tmin = computeMinThrustWorstCase(uav, N)
% COMPUTEMINTHRUSTWORSTCASE  Compute the worst-case minimum allowable thrust
%                            command over all angular rate references in
%                            [-wdmax, wdmax].
%
%   Tmin = computeMinThrustWorstCase(uav, N)
%
%   Evaluates computeMinThrustGivenWd over an N×N×N grid spanning the full
%   angular rate reference space and returns the maximum (worst-case) value.
%   This determines the minimum collective thrust T that guarantees physical
%   admissibility of the rotor speed equilibrium for any wd in the rate limit
%   box.
%
% INPUT
%   uav  - UAV parameter struct (see SETDEFAULTPARAMS) with fields:
%            .wdmaxx, .wdmaxy, .wdmaxz  angular rate limits  (rad/s)
%            (all fields required by COMPUTEMINTHRUSTGIVENWD)
%   N    - scalar grid resolution per axis (total evaluations: N^3)
%            recommended: N = 10 for a quick check, N = 50 for accuracy
%
% OUTPUT
%   Tmin - scalar worst-case minimum allowable collective thrust  (N)
%          Tmin = max_{wd in [-wdmax, wdmax]} computeMinThrustGivenWd(uav, wd)
%
% EXAMPLE
%   uav  = setDefaultParams();
%   Tmin = computeMinThrustWorstCase(uav, 20);
%   fprintf('Set T > %.4f to guarantee admissibility.\n', Tmin);
%
% See also: COMPUTEMINTHRUSTGIVENWD, SETDEFAULTPARAMS, CHECKGEOMETRY

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    %% --- Print search bounds --------------------------------------------
    fprintf('  Angular rate limits:\n');
    fprintf('    wdmaxx = %.4f rad/s  (%.2f deg/s)\n', uav.wdmaxx, rad2deg(uav.wdmaxx));
    fprintf('    wdmaxy = %.4f rad/s  (%.2f deg/s)\n', uav.wdmaxy, rad2deg(uav.wdmaxy));
    fprintf('    wdmaxz = %.4f rad/s  (%.2f deg/s)\n', uav.wdmaxz, rad2deg(uav.wdmaxz));
    fprintf('  Grid resolution: N = %d  (%d total evaluations)\n\n', N, N^3);

    %% --- Build 3D meshgrid ----------------------------------------------
    wx = linspace(-uav.wdmaxx, uav.wdmaxx, N);
    wy = linspace(-uav.wdmaxy, uav.wdmaxy, N);
    wz = linspace(-uav.wdmaxz, uav.wdmaxz, N);

    [WX, WY, WZ] = meshgrid(wx, wy, wz);

    %% --- Evaluate Tmin at every grid point ------------------------------
    Tmin_grid = zeros(size(WX));

    for i = 1:numel(WX)
        wd           = [WX(i); WY(i); WZ(i)];
        Tmin_grid(i) = computeMinThrustGivenWd(uav, wd);
    end

    %% --- Extract worst-case result --------------------------------------
    [Tmin, idx_max] = max(Tmin_grid(:));
    wd_worst        = [WX(idx_max); WY(idx_max); WZ(idx_max)];

    fprintf('  Worst-case result:\n');
    fprintf('    wd_worst = [% .4f, % .4f, % .4f] rad/s\n', ...
        wd_worst(1), wd_worst(2), wd_worst(3));
    fprintf('    Tmin     = %.6f\n\n', Tmin);

end