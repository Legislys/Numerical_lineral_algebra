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

    to_be_zeroed = [(R[(i+1)::, i], i) for i in range(n-1)]
    indices = [((i+1+np.array((abs(arg) >= tolerance).nonzero())), i)
               for arg, i in to_be_zeroed]
    rotations = []

    for indice, j in indices:
        for i in indice[0]:
            a = R[j, j]
            b = R[i, j]
            r = np.sqrt(a**2+b**2)
            c = a/r
            s = -b/r
            rotation = np.identity(n)
            rotation[j, j], rotation[i, i], rotation[i,j], rotation[j, i] = c, c, s, -s
            rotations.append(rotation)
            R = np.dot(rotation, R)
    Q = (np.linalg.multi_dot(rotations)).T
    return Q, R

