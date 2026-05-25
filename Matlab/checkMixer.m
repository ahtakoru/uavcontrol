function prob = checkMixer(uav)
% CheckMixer  Verify mandatory and reference properties of the
%                           UAV control-mixer matrix.
%
%   prob = checkMixer(uav)
%
%   Evaluates five mandatory properties (Properties 1–5) and eight optional
%   reference properties (Ref Properties 1–8) of the mixer matrix uav.M.
%   Mandatory failures set prob = 1; reference failures are informational.
%
% MIXER MATRIX SIGN STRUCTURE
%   uav.M = [-m11   m12   m13  m14 ;
%             m21  -m22   m23  m24 ;
%             m31   m32  -m33  m34 ;
%            -m41  -m42  -m43  m44 ];
%
%   where all mij scalars are non-negative by convention.
%
% INPUT
%   uav  - UAV parameter struct with fields:
%            .M              (4×4) mixer matrix (as structured above)
%            .m11–.m44       individual mixer coefficients (non-negative scalars)
%
% OUTPUT
%   prob - Scalar flag:  0 = all mandatory properties satisfied
%                        1 = one or more mandatory properties violated
%
% MANDATORY PROPERTIES
%   1. det(M) > 1e-3          M is nonsingular (invertible)
%   2. sum(M[:,1]) == 0       roll column sums to zero
%   3. sum(M[:,2]) == 0       pitch column sums to zero
%   4. sum(M[:,3]) == 0       yaw column sums to zero
%   5. sum(M[:,4]) > 0        thrust column sum is positive
%
% REFERENCE PROPERTIES (informational, do not affect prob)
%   1. det(M) > 1e-3
%   2. m11 == m21  and  m31 == m41
%   3. m12 == m22  and  m32 == m42
%   4. m13 + m23 - m33 - m43 == 0
%   5. m14 + m24 - m34 - m44 == 0
%   6. mk1 >= 0  and  mk2 >= 0  for k = 1,2,3,4
%   7. mk3 > 0  for k = 1,2,3,4
%   8. mk4 > 0  for k = 1,2,3,4
%
% EXAMPLE
%   prob = checkMixer(uav);
%   if prob, error('Mixer matrix violates mandatory properties.'); end
%
% See also: checkGeometry, computeLevel1EqPoint

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------
    fprintf('Static Mixer Algebraic Requirements Validation.\n');
    prob = 0;

    % Convenience: extract scalar coefficients once
    m11 = uav.m11;  m12 = uav.m12;  m13 = uav.m13;  m14 = uav.m14;
    m21 = uav.m21;  m22 = uav.m22;  m23 = uav.m23;  m24 = uav.m24;
    m31 = uav.m31;  m32 = uav.m32;  m33 = uav.m33;  m34 = uav.m34;
    m41 = uav.m41;  m42 = uav.m42;  m43 = uav.m43;  m44 = uav.m44;

    detM = det(uav.M);

    %% --- Mandatory Properties --------------------------------------------
    fprintf('=== Mandatory Properties ===\n');

    log_check(abs(detM) > 1e-3, 1, true, prob, ...
        'abs(det(M)) > 1e-3 — mixer matrix is nonsingular', ...
        'abs(det(M)) <= 1e-3 — mixer matrix is singular or ill-conditioned');
    if ~(abs(detM) > 1e-3), prob = 1; end

    col_tol = 1e-9;   % numerical tolerance for column-sum checks

    log_check(abs(sum(uav.M(:,1))) < col_tol, 2, true, prob, ...
        'sum(M[:,1]) == 0 — roll column balanced', ...
        'sum(M[:,1]) ~= 0 — roll column unbalanced');
    if ~(abs(sum(uav.M(:,1))) < col_tol), prob = 1; end

    log_check(abs(sum(uav.M(:,2))) < col_tol, 3, true, prob, ...
        'sum(M[:,2]) == 0 — pitch column balanced', ...
        'sum(M[:,2]) ~= 0 — pitch column unbalanced');
    if ~(abs(sum(uav.M(:,2))) < col_tol), prob = 1; end

    log_check(abs(sum(uav.M(:,3))) < col_tol, 4, true, prob, ...
        'sum(M[:,3]) == 0 — yaw column balanced', ...
        'sum(M[:,3]) ~= 0 — yaw column unbalanced');
    if ~(abs(sum(uav.M(:,3))) < col_tol), prob = 1; end

    log_check(sum(uav.M(:,4)) > 0, 5, true, prob, ...
        'sum(M[:,4]) > 0 — thrust column positive', ...
        'sum(M[:,4]) <= 0 — thrust column non-positive');
    if ~(sum(uav.M(:,4)) > 0), prob = 1; end

    %% --- Reference Properties (informational) ----------------------------
    fprintf('\n=== Reference Properties (informational) ===\n');

    log_check(abs(detM) > 1e-3, 1, false, prob, ...
        'det(M) > 1e-3 — mixer matrix is nonsingular', ...
        'det(M) <= 1e-3 — mixer matrix is singular or ill-conditioned');

    log_check(m11 == m21 && m31 == m41, 2, false, prob, ...
        'm11 == m21  and  m31 == m41', ...
        'm11 ~= m21  or   m31 ~= m41');

    log_check(m12 == m22 && m32 == m42, 3, false, prob, ...
        'm12 == m22  and  m32 == m42', ...
        'm12 ~= m22  or   m32 ~= m42');

    log_check(abs(m13 + m23 - m33 - m43) < col_tol, 4, false, prob, ...
        'm13 + m23 - m33 - m43 == 0', ...
        'm13 + m23 - m33 - m43 ~= 0');

    log_check(abs(m14 + m24 - m34 - m44) < col_tol, 5, false, prob, ...
        'm14 + m24 - m34 - m44 == 0', ...
        'm14 + m24 - m34 - m44 ~= 0');

    log_check(m11>=0 && m21>=0 && m31>=0 && m41>=0 && ...
              m12>=0 && m22>=0 && m32>=0 && m42>=0, 6, false, prob, ...
        'mk1 >= 0 and mk2 >= 0 for all k', ...
        'mk1 < 0  or  mk2 < 0  for some k');

    log_check(m13>0 && m23>0 && m33>0 && m43>0, 7, false, prob, ...
        'mk3 > 0 for all k', ...
        'mk3 <= 0 for some k');

    log_check(m14>0 && m24>0 && m34>0 && m44>0, 8, false, prob, ...
        'mk4 > 0 for all k', ...
        'mk4 <= 0 for some k');

    fprintf('\n');

end

% --------------------------------------------------------------------------
function log_check(condition, index, is_mandatory, ~, msg_pass, msg_fail)
% LOG_CHECK  Print a single PASS/FAIL/INFO line for one property check.
%
%   log_check(condition, index, is_mandatory, ~, msg_pass, msg_fail)
%
%   The fourth argument (~) is unused but kept so callers can pass `prob`
%   for readability at the call site.

    if is_mandatory
        prefix_pass = sprintf('  [PASS] Property %d: ', index);
        prefix_fail = sprintf('  [FAIL] Property %d: ', index);
    else
        prefix_pass = sprintf('  [PASS] Ref Property %d: ', index);
        prefix_fail = sprintf('  [FAIL] Ref Property %d: ', index);
    end

    if condition
        fprintf('%s%s\n', prefix_pass, msg_pass);
    else
        fprintf('%s%s\n', prefix_fail, msg_fail);
    end

end