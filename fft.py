import numpy as np

def geometric_progression(r, q, n):
    exponents = np.array(np.arange(n), ndmin=2)
    progression = r*np.array([q]*n, ndmin=2)**exponents
    return progression

def recursive_FFT(a):
    n = len(a)
    if n == 1:
        return a
    a_even = a[::2]
    a_odd = a[1::2]
    a_even = recursive_FFT(a_even)
    a_odd = recursive_FFT(a_odd)

    w = np.exp(-2j*np.pi/n)
    unity_progression = geometric_progression(1, w, n//2)
    a_hats = np.hstack((a_even + unity_progression*a_odd,
                       a_even - unity_progression*a_odd))
    return a_hats


def DFT(a):
    n = a.shape[1]
    w = np.exp(-2j*np.pi/n)
    n_th_roots_progression = geometric_progression(1, w, n)
    vandermonde = np.array(
        [n_th_roots_progression**k for k in range(n)], dtype='complex_')
    return np.dot(vandermonde.reshape(n, n), a.T).T


def inverse_recursive_FFT(a):
    n = a.shape[1]
    if n == 1:
        return a
    a_even = np.array([a[:, i] for i in range(0, n, 2)], dtype='complex_')
    a_odd = np.array([a[:, i] for i in range(1, n, 2)], dtype='complex_')
    a_even = recursive_FFT(a_even)
    a_odd = recursive_FFT(a_odd)
    w = np.exp(2j*np.pi/n)
    unity_progression = geometric_progression(1, w, n//2)
    a_hats = np.hstack((a_even + unity_progression*a_odd,
                       a_even - unity_progression*a_odd))
    return a_hats


def inverse_DFT(a):
    n = a.shape[1]
    w = np.exp(2j*np.pi/n)
    n_th_roots_progression = geometric_progression(1, w, n)
    vandermonde = np.array(
        [n_th_roots_progression**k for k in range(n)], dtype='complex_')
    return np.dot(vandermonde.reshape(n, n), a.T).T
