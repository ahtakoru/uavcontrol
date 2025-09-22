import numpy as np
import eq_point

def determine_thrust_min_threshold(uav, T_low=0.01, T_high=1.0, tol=1e-9, max_iter=10000):
    """
    Uses the bisection method to find T_min such that when T = T_min, 
    at least one component of wr_ast is zero.

    Parameters:
    - self: The object that contains `compute_equilibrium_point`
    - uav: The UAV object containing wmaxx, wmaxy, wmaxz
    - T_low: Lower bound of the search interval (default: 0.0)
    - T_high: Upper bound of the search interval (default: 1.0)
    - tol: Convergence tolerance for bisection (default: 1e-6)
    - max_iter: Maximum iterations to prevent infinite loops (default: 100)

    Returns:
    - T_min: The estimated minimum T such that at least one component of wr_ast is zero.
    """
    
    def has_zero_component(T):
        """ Helper function to check if wr_ast has any zero component """
        _, wr_ast, _ = eq_point.compute_equilibrium_point(uav, T, [uav.wmaxx, uav.wmaxy, 0])
        return any(w < tol for w in wr_ast)


    # Ensure T_low is in the region where at least one component is zero
    if not has_zero_component(T_low):
        raise ValueError("T_low does not have a nonpositive component. Adjust the range.")

    # Ensure T_high is in the region where all components are positive
    if has_zero_component(T_high):
        raise ValueError("T_high still has a nonpositive component. Adjust the range.")

    # Bisection method
    for _ in range(max_iter):
        T_mid = (T_low + T_high) / 2.0

        if has_zero_component(T_mid):
            T_low = T_mid  # Move lower bound up
        else:
            T_high = T_mid  # Move upper bound down
        
        if abs(T_high - T_low) < tol:
            break  # Converged

    return (T_low + T_high) / 2.0  # Return the midpoint as the estimated T_min