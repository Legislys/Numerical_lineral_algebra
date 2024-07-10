import numpy as np
from QR_decomposition import QR_givens_rotations

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

# More optimized code using Gauss elimination algorithm

def determinant(B):
    A = B.copy()
    if A.shape[0] != A.shape[1]:
        return f'{A} is non-square matrix'
    n = A.shape[0]
    tolerance = 1e-06
    for i in range(n):
        highest_column_coef = i + np.argmax(abs(A[i:n+1, i]))
        if abs(A[highest_column_coef, i]) < tolerance:
            continue
        A[[i, highest_column_coef]] = A[[highest_column_coef, i]]

        for j in range(i+1, n):
            A[j, :] = A[j, :] - (A[j, i]/A[i, i])*A[i, :]
    return np.prod(np.diag(A))

def QR_determinant(B):
    Q,R = QR_givens_rotations(B)
    return np.prod(np.diag(R))