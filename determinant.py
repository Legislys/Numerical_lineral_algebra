import numpy as np

# Implentation of Laplace formula, algorithmically expenesive

def Laplace_determinant(B):
    A = B.copy()
    n = A.shape[1]
    if n == 1:
        return A[0, 0]
    determinant = 0
    for i in range(n):
        minor = np.delete(np.delete(A, 0, 0), i, 1)
        cofactor = ((-1) ** i) * A[0, i] * Laplace_determinant(minor)
        determinant += cofactor
    return determinant