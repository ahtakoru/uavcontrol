classdef QuaternionTools
% QUATERNIONTOOLS  Static utility class for unit-quaternion arithmetic and
%                  related geometric operations.
%
% CONVENTION
%   Quaternions are represented as (4×1) column vectors:
%     q = [q0; q1; q2; q3]
%   where q0 is the scalar (real) part and q1, q2, q3 form the vector
%   (imaginary) part. Unit quaternions satisfy norm(q) = 1.
%
% METHODS
%   QuaternionTools.rotationMatrix(q)    — quaternion → (3×3) rotation matrix
%   QuaternionTools.multiplication(q, p) — Hamilton product  q ⊗ p
%   QuaternionTools.conjugate(q)         — quaternion conjugate  q*
%   QuaternionTools.norm(q)              — quaternion norm  ||q||
%   QuaternionTools.inverse(q)           — quaternion inverse  q^{-1}
%   QuaternionTools.imag(q)              — extract vector part  [q1; q2; q3]
%   QuaternionTools.embed(w)             — embed 3-vector as pure quaternion  [0; w]
%   QuaternionTools.sign(q0)             — sign of scalar part  (±1)
%   QuaternionTools.skew(v)              — (3×3) skew-symmetric matrix of v
%
% EXAMPLE
%   q = [1; 0; 0; 0];                        % identity quaternion
%   R = QuaternionTools.rotationMatrix(q);    % should return eye(3)
%   p = QuaternionTools.multiplication(q, q); % should return q
%
% REFERENCE
%   Shuster, M. D. (1993). A survey of attitude representations.
%   Journal of the Astronautical Sciences, 41(4), 439–517.
%
% See also: COMPUTEEQPOINT, LEVEL1COMPACTFORMS, SETDEFAULTPARAMS

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    methods (Static)

        function R = rotationMatrix(q)
        % ROTATIONMATRIX  Convert a unit quaternion to a rotation matrix.
        %
        %   R = QuaternionTools.rotationMatrix(q)
        %
        %   INPUT
        %     q  - (4×1) unit quaternion  [q0; q1; q2; q3]
        %
        %   OUTPUT
        %     R  - (3×3) orthogonal rotation matrix  (R'*R = I, det(R) = 1)

            q = q(:);
            q0 = q(1);  q1 = q(2);  q2 = q(3);  q3 = q(4);

            R = [ 1 - 2*(q2^2 + q3^2),   2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2) ;
                    2*(q1*q2 + q0*q3), 1 - 2*(q1^2 + q3^2),   2*(q2*q3 - q0*q1) ;
                    2*(q1*q3 - q0*q2),   2*(q2*q3 + q0*q1), 1 - 2*(q1^2 + q2^2) ];
        end

        function y = multiplication(q, p)
        % MULTIPLICATION  Hamilton product of two quaternions  y = q ⊗ p.
        %
        %   y = QuaternionTools.multiplication(q, p)
        %
        %   INPUT
        %     q, p  - (4×1) quaternions  [q0; q1; q2; q3]
        %
        %   OUTPUT
        %     y     - (4×1) quaternion product  q ⊗ p

            y = zeros(size(q), 'like', q);
            y(1) = q(1)*p(1) - q(2)*p(2) - q(3)*p(3) - q(4)*p(4);
            y(2) = q(1)*p(2) + q(2)*p(1) + q(3)*p(4) - q(4)*p(3);
            y(3) = q(1)*p(3) - q(2)*p(4) + q(3)*p(1) + q(4)*p(2);
            y(4) = q(1)*p(4) + q(2)*p(3) - q(3)*p(2) + q(4)*p(1);
        end

        function qBar = conjugate(q)
        % CONJUGATE  Quaternion conjugate  q* = [q0; -q1; -q2; -q3].
        %
        %   qBar = QuaternionTools.conjugate(q)
        %
        %   INPUT
        %     q     - (4×1) quaternion
        %
        %   OUTPUT
        %     qBar  - (4×1) conjugate quaternion

            qBar      =  q(:);
            qBar(2:4) = -qBar(2:4);
        end

        function n = norm(q)
        % NORM  Euclidean norm of a quaternion  ||q|| = sqrt(q0²+q1²+q2²+q3²).
        %
        %   n = QuaternionTools.norm(q)
        %
        %   INPUT
        %     q  - (4×1) quaternion
        %
        %   OUTPUT
        %     n  - non-negative scalar norm

            n = sqrt(sum(q(:).^2));
        end

        function qInv = inverse(q)
        % INVERSE  Quaternion inverse  q^{-1} = q* / ||q||².
        %
        %   qInv = QuaternionTools.inverse(q)
        %
        %   For unit quaternions  q^{-1} = q*  (conjugate suffices).
        %
        %   INPUT
        %     q     - (4×1) quaternion  (non-zero norm)
        %
        %   OUTPUT
        %     qInv  - (4×1) inverse quaternion

            qInv = QuaternionTools.conjugate(q) / QuaternionTools.norm(q)^2;
        end

        function qImag = imag(q)
        % IMAG  Extract the vector (imaginary) part of a quaternion.
        %
        %   qImag = QuaternionTools.imag(q)
        %
        %   INPUT
        %     q      - (4×1) quaternion  [q0; q1; q2; q3]
        %
        %   OUTPUT
        %     qImag  - (3×1) vector part  [q1; q2; q3]

            qImag = q(2:4);
        end

        function qEmbed = embed(w)
        % EMBED  Embed a 3-vector as a pure quaternion  [0; w].
        %
        %   qEmbed = QuaternionTools.embed(w)
        %
        %   INPUT
        %     w       - (3×1) vector
        %
        %   OUTPUT
        %     qEmbed  - (4×1) pure quaternion  [0; w1; w2; w3]

            qEmbed = [0; w(:)];
        end

        function s = sign(q0)
        % SIGN  Sign of the scalar part of a quaternion  (±1).
        %
        %   s = QuaternionTools.sign(q0)
        %
        %   Used to enforce the canonical hemisphere convention (q0 >= 0),
        %   which ensures a unique quaternion representation per rotation.
        %
        %   INPUT
        %     q0  - scalar part of a quaternion
        %
        %   OUTPUT
        %     s   - +1 if q0 >= 0, -1 if q0 < 0

            if q0 < 0
                s = -1;
            else
                s =  1;
            end
        end

        function S = skew(v)
        % SKEW  Skew-symmetric (cross-product) matrix of a 3-vector.
        %
        %   S = QuaternionTools.skew(v)
        %
        %   For any vector u:  S*u = cross(v, u)
        %
        %   INPUT
        %     v  - (3×1) vector  [v1; v2; v3]
        %
        %   OUTPUT
        %     S  - (3×3) skew-symmetric matrix:
        %            [  0  -v3   v2 ]
        %            [ v3    0  -v1 ]
        %            [-v2   v1    0 ]

            S = [    0, -v(3),  v(2) ;
                  v(3),     0, -v(1) ;
                 -v(2),  v(1),     0 ];
        end

    end
end