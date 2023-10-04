import scipy as sp

from scipy.sparse import hstack

'''
Given a linear space A, proejct A to its closest linear space that satisfies
the constraints AB = 0
'''
def closest_orthogonal_subspace(A, B, M=None, mass_orthogonal=False):
    if M is None:
        M = sp.sparse.identity(A.shape[0])
    c = B.shape[0]

    BMA = B.T @ M @ A

    if mass_orthogonal:
        A2 = A - B @ BMA
    else:
        BMB = B.T @ M @ B
        BMBBMA = sp.sparse.linalg.spsolve(BMB, BMA)
        A2 = A - B @ BMBBMA
    return A2

