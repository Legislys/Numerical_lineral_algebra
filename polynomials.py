# Straightforward implementation

def multiply_polynomials(x, y):
    n, m = len(x), len(y)
    result = [0]*(n+m-1)
    for i in range(n):
        for j in range(m):
            result[i+j] += x[i]*y[j]
    return result

