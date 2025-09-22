import numpy as np

def multiply(q, p):
    q0, q1, q2, q3 = q
    M = np.array([
                  [q0, -q1, -q2, -q3],
                  [q1,  q0, -q3,  q2],
                  [q2,  q3,  q0, -q1], 
                  [q3, -q2,  q1,  q0],
                ])
    
    return np.matmul(M, p)

def inverse(q):
    q0, q1, q2, q3 = q
    return np.array([q0, -q1, -q2, -q3])/np.linalg.norm(q)

def normalize(q):
    return q/np.linalg.norm(q)

def imag(q):
    return np.array([q[1], q[2], q[3]])

def embed_vec(w):
    return np.array([0, w[0], w[1], w[2]])

def sign(r):
    if r < 0: return -1.
    else: return 1.

