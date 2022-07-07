import numpy as np
from scipy import linalg
import math


def generate_random_matrix(n):
    return np.random.random((n, n))


def compute_eigvals(matrix):
    return np.linalg.eigvals(matrix)


def is_spd(matrix):
    return np.allclose(matrix, matrix.transpose()) and np.all(compute_eigvals(matrix) > 0)


def generate_spd_matrix(n):
    rand_matrix = generate_random_matrix(n)
    spsd_matrix = np.matmul(rand_matrix.transpose(), rand_matrix)
    u, _, vt = np.linalg.svd(spsd_matrix)
    spd_matrix = np.matmul(np.matmul(u, 1 + np.diag(np.random.random(n))), vt)

    return spd_matrix


def compute_square_root(matrix):
    return linalg.sqrtm(matrix)


def compute_d_bw(a, b):
    subtraction = 2 * compute_square_root(np.matmul(np.matmul(compute_square_root(a), b), compute_square_root(a)))

    return math.sqrt(np.trace(a + b - subtraction))


def compute_d_2(a, b):
    ab = np.matmul(a, b)
    ab_eigvals = compute_eigvals(ab)

    sum_eigval_square_root = 0

    for eigval in ab_eigvals:
        sum_eigval_square_root += math.sqrt(eigval)

    subtraction = 2 * sum_eigval_square_root

    return math.sqrt(np.trace(a) + np.trace(b) - subtraction)


if __name__ == '__main__':
    n = 5

    matrix = generate_spd_matrix(n)
    print('Candidate matrix:')
    print(matrix)
    print('Is SPD:', is_spd(matrix))

    a = generate_spd_matrix(n)
    b = generate_spd_matrix(n)

    print('a:')
    print(a)

    print('b:')
    print(b)

    print('2.2.1. d_bw:', compute_d_bw(a, b))
    print('2.2.2. d_2:', compute_d_2(a, b))
