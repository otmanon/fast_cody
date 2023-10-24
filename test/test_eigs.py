import igl

from .context import fast_cd as fcd
from .context import fast_cd_pyb as fcdp
from .context import cvxopt
from .context import unittest
from .context import numpy as np
from .context import scipy as sp
class TestEigs(unittest.TestCase):

    def test_umfpack_lu_factorization_indefinite(self):
        msh_file = fcd.get_data('cd_fish.msh')
        [V, F, T] = fcdp.readMSH(msh_file)
        W = np.ones((V.shape[0], 1))
        J = fcd.lbs_jacobian(V, W)
        C = fcd.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fcd.lbs_weight_space_constraint(V, C)
        # [B, l, Ws] = fcd.skinning_subspace(V, T, 10, num_clusters, C=C2, read_cache=read_cache,

        t = 1e5
        for i in range(10):
            random = (np.random.randn(V.shape[0], V.shape[1]) - 0.5) * 2
            U = V + random * t
            L = fcd.laplacian(U, T)
            c = C2.shape[0]
            Z = sp.sparse.csc_matrix((c, c))
            A = sp.sparse.vstack((sp.sparse.hstack((L, C2.T)),
                        sp.sparse.hstack((C2, Z)))).tocsc()


            [I, J] = A.nonzero()
            v = A.data

            Ac = cvxopt.spmatrix(v, I, J, A.shape)
            F = cvxopt.umfpack.symbolic(Ac)
            numeric = cvxopt.umfpack.numeric(Ac, F)
            b = cvxopt.matrix(np.zeros(A.shape[0]))
            cvxopt.umfpack.solve(Ac, numeric, b)


    def test_umfpack_lu_factorization_psd(self):
        msh_file = fcd.get_data('cd_fish.msh')
        [V, F, T] = fcdp.readMSH(msh_file)

        t = 1e10
        for i in range(10):
            random = (np.random.randn(V.shape[0], V.shape[1])-0.5)*2
            U = V + random*t
            A = fcd.laplacian(U, T)
            [I, J] = A.nonzero()
            v = A.data


            Ac = cvxopt.spmatrix(v, I, J, A.shape)
            F = cvxopt.umfpack.symbolic(Ac)
            numeric = cvxopt.umfpack.numeric(Ac, F)
            b = cvxopt.matrix(np.zeros(A.shape[0]))
            cvxopt.umfpack.solve(Ac, numeric, b)


    def test_psd(self):

        msh_file = fcd.get_data('cd_fish.msh')
        [V, F, T] = fcdp.readMSH(msh_file)

        L = fcd.laplacian(V, T)
        M = igl.massmatrix(V, T)

        [D, B] = fcd.eigs(L, 10, M)

        t = 1e-12
        # print(D.real)
        # first eigenvalue should be zero
        self.assertTrue(np.alltrue(D[0:1] < t))

        # first eigenvector should be constant
        self.assertTrue(np.std(B[:, 0]) < t)






if __name__ == '__main__':
    unittest.main()