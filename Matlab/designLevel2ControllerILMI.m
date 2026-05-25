function K = designLevel2ControllerILMI(A, B, C, Q, isStructured)
% DESIGNLEVEL2CONTROLLERILMI  Iterative LMI solver for Static Output-Feedback 
% (SOF) stabilization.
%
%   K = designLevel2ControllerILMI(A, B, C, Q, isStructured)
%
%   Solves the static output-feedback problem
%
%       u = -K y,   y = C x,   dx/dt = A x + B u
%
%   using an iterative LMI (iLMI) algorithm.  At each iteration the
%   algorithm (i) minimizes alpha (the closed-loop decay-rate margin) via
%   bisection over an LMI feasibility problem, and (ii) updates the
%   Lyapunov variable X by minimizing trace(P) subject to the same LMI.
%   Convergence is declared when alpha < 0 (stabilizing K found) or when
%   X stops changing (no solution available).
%
% INPUTS
%   A            - (n×n) system matrix
%   B            - (n×m) input matrix
%   C            - (p×n) output matrix
%   Q            - (n×n) positive-definite weight for the CARE initialization
%   isStructured - logical flag:
%                    true  → K is diagonal (3x3) structure
%                    false → K is a full (3x3) matrix
%
% OUTPUT
%   K  - (m×p) or structured (3×9) static output-feedback gain matrix
%        Returns 0 if no stabilizing gain is found.
%
% ALGORITHM
%   Step 0 — Initialize X via the solution of an algebraic Riccati equation.
%   Step 1 — Minimize alpha subject to the SOF-LMI  (bisection search).
%   Step 2 — If alpha < 0, return K; otherwise update X = argmin trace(P).
%   Step 3 — Repeat until convergence or stagnation.
%
% DEPENDENCIES
%   YALMIP (sdpvar, optimize, sdpsettings, value)
%   Control System Toolbox (icare)
%
% EXAMPLE
%   K = designLevel2ControllerILMI(A, B, C, eye(n), true);
%
% REFERENCE
%   Cao, Y.-Y., Lam, J., & Sun, Y.-X. (1998). Static output feedback
%   stabilization: An ILMI approach. Automatica, 34(12), 1641–1645.
%
% See also: LEVEL1_COMPACT_FORMS, COMPUTE_EQ_POINT

% --------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% --------------------------------------------------------------------------

    K = 0;

    %% --- Step 0: Initialize X via CARE ----------------------------------
    X = icare(A, [], Q, [], [], [], -B*B');
    X = (X + X') / 2;   % enforce symmetry

    %% --- Main iLMI loop -------------------------------------------------
    while true

        % Step 1 — minimize alpha (bisection over LMI feasibility)
        [alpha, ~, K] = minimize_alpha(A, B, C, X, isStructured);
        fprintf('  [iLMI] alpha = %.6f\n', alpha);

        if alpha < 0
            fprintf('  [iLMI] Stabilizing K found (alpha = %.6f).\n', alpha);
            break;
        end

        % Step 2 — update X by minimizing trace(P)
        [~, P_star] = minimize_trace(A, B, C, X, alpha, isStructured);

        if norm(X - P_star, 2) < 1e-3
            fprintf('  [iLMI] Stagnation detected — no solution available.\n');
            K = 0;
            break;
        end

        X = P_star;
    end

end


% ==========================================================================
function [alpha, P, K] = minimize_alpha(A, B, C, X, isStructured)
% MINIMIZE_ALPHA  Find the minimum feasible alpha via bisection.
%
%   Bisects over [alpha_min, alpha_max] and calls SOLVE_LMI at each
%   midpoint.  Returns the smallest alpha for which the LMI is feasible,
%   together with the corresponding P and K.

    alpha_min = -1000;
    alpha_max =  1000;
    tol       =  1e-6;
    max_iter  =  100000;

    % Verify that the upper bound is feasible before bisecting
    [prob_max, P_best, K_best] = solve_LMI(A, B, C, X, alpha_max, isStructured);
    if prob_max == 1
        error('designLevel2ControllerILMI:infeasible', ...
            'No feasible alpha found in [%.4g, %.4g].', alpha_min, alpha_max);
    end

    for iter = 1:max_iter
        alpha_mid = (alpha_min + alpha_max) / 2;
        [prob_mid, P_mid, K_mid] = solve_LMI(A, B, C, X, alpha_mid, isStructured);

        if prob_mid == 0          % feasible: tighten upper bound
            alpha_max = alpha_mid;
            P_best    = P_mid;
            K_best    = K_mid;
        else                      % infeasible: raise lower bound
            alpha_min = alpha_mid;
        end

        if abs(alpha_max - alpha_min) < tol
            break
        end
    end

    alpha = alpha_max;
    P     = P_best;
    K     = K_best;
end


% ==========================================================================
function [prob, P, K] = solve_LMI(A, B, C, X, alpha, isStructured)
% SOLVE_LMI  LMI feasibility problem for a given alpha and Lyapunov seed X.
%
%   Solves:
%       find   P > 0, K
%       s.t.   [A'P + PA - XBB'P - (XBB'P)' + X'BB'X - αP,  (B'P + KC)' ]
%              [                    B'P + KC,                    -I        ] <= 0

    n = size(A, 2);
    m = size(B, 2);

    P   = sdpvar(n, n, 'symmetric');
    K   = build_K(m, size(C,1), isStructured);

    LMI = build_LMI(A, B, C, X, alpha, P, K);

    Fset = [P >= 1e-12*eye(n)] + [LMI <= -1e-16*eye(n + m)];
    ops  = sdpsettings('verbose', 0);
    optimize(Fset, [], ops);

    [prob, P, K] = extract_and_check(P, K, LMI, n, m);
end


% ==========================================================================
function [prob, P] = minimize_trace(A, B, C, X, alpha, isStructured)
% MINIMIZE_TRACE  Minimize trace(P) subject to the SOF-LMI at fixed alpha.

    n = size(A, 2);
    m = size(B, 2);

    P = sdpvar(n, n, 'symmetric');
    K = build_K(m, size(C,1), isStructured);

    LMI = build_LMI(A, B, C, X, alpha, P, K);

    Fset = [P >= 1e-12*eye(n)] + [LMI <= -1e-16*eye(n + m)];
    ops  = sdpsettings('verbose', 0);
    optimize(Fset, trace(P), ops);

    [prob, P, ~] = extract_and_check(P, K, LMI, n, m);
end


% ==========================================================================
%  Shared helpers
% ==========================================================================

function K = build_K(m, p, isStructured)
% BUILD_K  Construct the YALMIP decision variable for K.
%
%   isStructured = true  → block-diagonal PID structure [diag(Kp), diag(Ki), diag(Kd)]
%   isStructured = false → full (m×p) matrix

    if isStructured
        sdpvar kp_q1 kp_q2 kp_q3
        K = diag([kp_q1, kp_q2, kp_q3]);
    else
        K = sdpvar(m, p, 'full');
    end
end


function LMI = build_LMI(A, B, C, X, alpha, P, K)
% BUILD_LMI  Assemble the SOF-LMI block matrix.

    m     = size(B, 2);
    LMI11 = A'*P + P*A - X*B*B'*P - (X*B*B'*P)' + X'*B*B'*X - alpha*P;
    LMI12 = (B'*P + K*C)';
    LMI22 = -eye(m);
    LMI   = [ LMI11,  LMI12  ;
              LMI12', LMI22  ];
end


function [prob, P, K] = extract_and_check(P_sdp, K_sdp, LMI_sdp, n, m)
% EXTRACT_AND_CHECK  Extract YALMIP values and verify positive-definiteness.

    P   = value(P_sdp);
    K   = value(K_sdp);
    LMI = value(LMI_sdp);

    prob = 0;
    if min(eig(P))   <= 0,  prob = 1;  end
    if max(eig(LMI)) >= 0,  prob = 1;  end
end