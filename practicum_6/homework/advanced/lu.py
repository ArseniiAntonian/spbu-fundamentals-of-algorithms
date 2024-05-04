from collections import defaultdict
from dataclasses import dataclass
import os
import yaml
import time

import numpy as np
from numpy.typing import NDArray
import scipy.io
import scipy.linalg

#from src.linalg import get_scipy_solution

def get_scipy_solution(A, b):
    lu_and_piv = scipy.linalg.lu_factor(A)
    return scipy.linalg.lu_solve(lu_and_piv, b)

@dataclass
class Performance:
    time: float = 0.0
    relative_error: float = 0.0


def lu(A: NDArray, permute: bool) -> tuple[NDArray, NDArray, NDArray]:
    n = len(A)
    U = np.copy(A)
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n - 1):
        if permute:
            pivot_index = np.argmax(np.abs(U[k:, k])) + k
            if pivot_index != k:
                U[[k, pivot_index]] = U[[pivot_index, k]]
                P[[k, pivot_index]] = P[[pivot_index, k]]
                if k > 0:
                    L[[k, pivot_index], :k] = L[[pivot_index, k], :k]

            L[k+1:, k] = U[k+1:, k] / U[k, k]
            U[k+1:, k:] -= np.outer(L[k+1:, k], U[k, k:])
        else:
            U_0 = np.eye(n)
            for i in range(k + 1, n):
                m = U[i, k] / U[k, k]
                U_0[i, k] = -m
                L[i, k] = m
            U = U_0 @ U

    return L, U, P


def solve(L: NDArray, U: NDArray, P: NDArray, b: NDArray) -> NDArray:

    b = P @ b
    
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x


def run_test_cases(n_runs: int, path_to_homework: str) -> dict[str, Performance]:
    matrix_filenames = []
    performance_by_matrix = defaultdict(Performance)
    with open(os.path.join(path_to_homework, "matrices.yaml"), "r") as f:
        matrix_filenames = yaml.safe_load(f)
    for i, matrix_filename in enumerate(matrix_filenames):
        print(f"Processing matrix {i+1} out of {len(matrix_filenames)}")
        A = (
            scipy.io.mmread(os.path.join(path_to_homework, "matrices", matrix_filename))
            .todense()
            .A
        )
        b = np.ones((A.shape[0],))
        perf = performance_by_matrix[matrix_filename]
        for j in range(n_runs):
            t1 = time.time()
            L, U, P = lu(A, permute=True)
            t2 = time.time()
            perf.time += t2 - t1
            if j == 0:  
                x = solve(L, U, P, b)
                x_exact = get_scipy_solution(A, b)
                perf.relative_error = np.linalg.norm(x - x_exact) / np.linalg.norm(
                    x_exact
                )
    return performance_by_matrix


if __name__ == "__main__":
    n_runs = 10
    path_to_homework = os.path.join("practicum_6", "homework", "advanced")
    performance_by_matrix = run_test_cases(
        n_runs=n_runs, path_to_homework=path_to_homework
    )

    print("\nResult summary:")
    for filename, perf in performance_by_matrix.items():
        print(
            f"Matrix: {filename}. "
            f"Average time: {perf.time / n_runs:.2e} seconds. "
            f"Relative error: {perf.relative_error:.2e}"
        )
