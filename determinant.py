import numpy as np

# Implentation of Laplace formula, algorithmically expenesive

def Laplace_det(A):
    if A.shape[0] != A.shape[1]:
        return f'{A} is non-square matrix'
    n = A.shape[0]
    if n == 1:
        return A[0, 0]
    B = np.delete(A, 0, 1)
    minors = np.array([np.delete(B, i, 0) for i in range(n)], ndmin=2)
    signs = (-1)**np.arange(n)
    return np.sum(signs * A[0, :] * np.array([Laplace_det(minor) for minor in minors]))