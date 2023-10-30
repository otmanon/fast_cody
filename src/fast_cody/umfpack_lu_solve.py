import numpy as np
import cvxopt
import cvxopt.umfpack

def umfpack_lu_solve(A, b):
    [I, J] = A.nonzero()
    v = A.data
    Ac= cvxopt.spmatrix(v, I, J, A.shape)
    bc = cvxopt.matrix(b)
    cvxopt.umfpack.linsolve(Ac, bc)
    cnp = np.array(bc)
    return cnp