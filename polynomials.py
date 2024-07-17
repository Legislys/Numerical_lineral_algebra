from fft import recursive_FFT, inverse_recursive_FFT

# Straightforward implementation


def multiply_polynomials(x, y):
    n, m = len(x), len(y)
    result = [0]*(n+m-1)
    for i in range(n):
        for j in range(m):
            result[i+j] += x[i]*y[j]
    return result


def FFT_multiply_polynomials(x, y):
    n, m = len(x), len(y)
    a = x+[0]*n
    b = y+[0]*m
    A = recursive_FFT(a)*recursive_FFT(b)
    return inverse_recursive_FFT(A)/(n+m)
