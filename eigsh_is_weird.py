import numpy as np
import scipy

eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
    np.array([[1., - 0.33333331, - 0.33333331, - 0.33333331],
              [- 0.33333331,  1., -0.33333331, - 0.33333331],
              [- 0.33333331, - 0.33333331,  1., - 0.33333331],
              [-0.33333331, - 0.33333331, - 0.33333331,  1.]]),
    3,
    which="SM",
    ncv=7,
    tol=1e-4,
    v0=np.ones(4),
    maxiter=7532 * 5,
    )
np.random.seed(42)
print(eigenvectors)
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
    np.array([[1., - 0.33333331, - 0.33333331, - 0.33333331],
              [- 0.33333331, 1., -0.33333331, - 0.33333331],
              [- 0.33333331, - 0.33333331, 1., - 0.33333331],
              [-0.33333331, - 0.33333331, - 0.33333331, 1.]]),
    3,
    which="SM",
    ncv=7,
    tol=1e-4,
    v0=np.random.rand(4),
    maxiter=7532 * 5,
    )
print(eigenvectors)
