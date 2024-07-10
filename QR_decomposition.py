import numpy as np

tolerance = 1e-06

# Straightforward implementation of QR decomposition using Givens rotations

def QR_givens_rotations(B):
    R = B.copy()
    if R.shape[0] != R.shape[1]:
        return f'{R} is non-square matrix.'

    n = R.shape[0]
    if n == 1:
        return R
    rotations = []

    for j in range(n - 1):
        for i in range(j + 1, n):
            if abs(R[i, j]) >= tolerance:
                a = R[j, j]
                b = R[i, j]
                r = np.hypot(a, b)
                c = a / r
                s = -b / r
                rotation = np.identity(n)
                rotation[j, j], rotation[i, i], rotation[i,j], rotation[j, i] = c, c, s, -s
                rotations.append(rotation)
                R = np.dot(rotation, R)
    Q = (np.linalg.multi_dot(rotations)).T
    return Q, R