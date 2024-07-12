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
    n = len(x)
    a = x+[0]*n
    b = y+[0]*n
    A = recursive_FFT(a)*recursive_FFT(b)
    return inverse_recursive_FFT(A)/(2*n)
