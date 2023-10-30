import igl

from .context import fast_cody as fcd
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

        t = 0
        for i in range(1):
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

        t = 0
        for i in range(1):
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


    def test_eigs_indefinite(self):
        msh_file = fcd.get_data('cd_fish.msh')
        [V, F, T] = fcdp.readMSH(msh_file)
        W = np.ones((V.shape[0], 1))
        J = fcd.lbs_jacobian(V, W)
        C = fcd.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fcd.lbs_weight_space_constraint(V, C)
        c = C2.shape[0]
        Z = sp.sparse.csc_matrix((c, c))
        M = igl.massmatrix(V, T)

        L = fcd.laplacian(V, T)
        A = sp.sparse.vstack((sp.sparse.hstack((L, C2.T)),
                              sp.sparse.hstack((C2, Z)))).tocsc()
        M = sp.sparse.block_diag((M, Z)).tocsc()

        threshold = 1e-12
        [E, B] = fcd.eigs(A, M=M, k=10)  # sp.sparse.linalg.eigs(L, M=M, k=num_modes, sigma=0, which='LM')
        B = B.real[0:L.shape[0], :]
        self.assertTrue(np.alltrue( C2 @ B < threshold ))
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

    def test_cd_skinning_subspace(self):
        msh_file = fcd.get_data('cd_fish.msh')
        [V, F, T] = fcdp.readMSH(msh_file)
        [V, so, to] = fcdp.scale_and_center_geometry(V, np.array(1), np.array([[0, 0, 0.]]))  # center to unit height and about origin

        W = np.ones((V.shape[0], 1))
        J = fcd.lbs_jacobian(V, W)
        C = fcd.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fcd.lbs_weight_space_constraint(V, C)

        try:
            [B, l, Ws] = fcd.skinning_subspace(V, T, 16, 1000, C=C2, read_cache=False,
                                              cache_dir=None, constraint_enforcement="optimal");
        except:
            self.assertTrue(False)

        try:
            [B2, l2, Ws2] = fcd.skinning_subspace(V, T, 16, 1000, C=C2, read_cache=False,
                                             cache_dir=None, constraint_enforcement="project");
        except:
            self.assertTrue(False)
    def test_laplacian_eigenmodes(self):
        msh_file = fcd.get_data('cd_fish.msh')
        [V, F, T] = fcdp.readMSH(msh_file)
        [V, so, to] = fcdp.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))  # center to unit height and about origin

        W = np.ones((V.shape[0], 1))
        J = fcd.lbs_jacobian(V, W)
        C = fcd.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fcd.lbs_weight_space_constraint(V, C)
        [W, E] = fcd.laplacian_eigenmodes(V, T, 16, read_cache=False, mu=1, J=C2,
                                      constraint_enforcement='project')
        [W2, E2] = fcd.laplacian_eigenmodes(V, T, 16, read_cache=False, mu=1, J=C2,
                                      constraint_enforcement='optimal')


if __name__ == '__main__':
    unittest.main()

# import igl
# import fast_cd as fcd
# import fast_cd_pyb as fcdp
# import cvxopt
# import numpy as np
# import scipy as sp
# def test_umfpack_lu_factorization_indefinite():
#     msh_file = fcd.get_data('cd_fish.msh')
#     [V, F, T] = fcdp.readMSH(msh_file)
#     W = np.ones((V.shape[0], 1))
#     J = fcd.lbs_jacobian(V, W)
#     C = fcd.complementary_constraint_matrix(V, T, J, dt=1e-3)
#     C2 = fcd.lbs_weight_space_constraint(V, C)
#     # [B, l, Ws] = fcd.skinning_subspace(V, T, 10, num_clusters, C=C2, read_cache=read_cache,
#
#     t = 1e5
#     for i in range(10):
#         random = (np.random.randn(V.shape[0], V.shape[1]) - 0.5) * 2
#         U = V + random * t
#         L = fcd.laplacian(U, T)
#         c = C2.shape[0]
#         Z = sp.sparse.csc_matrix((c, c))
#         A = sp.sparse.vstack((sp.sparse.hstack((L, C2.T)),
#                     sp.sparse.hstack((C2, Z)))).tocsc()
#
#
#         [I, J] = A.nonzero()
#         v = A.data
#
#         Ac = cvxopt.spmatrix(v, I, J, A.shape)
#         F = cvxopt.umfpack.symbolic(Ac)
#         numeric = cvxopt.umfpack.numeric(Ac, F)
#         b = cvxopt.matrix(np.zeros(A.shape[0]))
#         cvxopt.umfpack.solve(Ac, numeric, b)
#
#
# def test_umfpack_lu_factorization_psd():
#     msh_file = fcd.get_data('cd_fish.msh')
#     [V, F, T] = fcdp.readMSH(msh_file)
#
#     t = 1e10
#     for i in range(10):
#         random = (np.random.randn(V.shape[0], V.shape[1])-0.5)*2
#         U = V + random*t
#         A = fcd.laplacian(U, T)
#         [I, J] = A.nonzero()
#         v = A.data
#
#
#         Ac = cvxopt.spmatrix(v, I, J, A.shape)
#         F = cvxopt.umfpack.symbolic(Ac)
#         numeric = cvxopt.umfpack.numeric(Ac, F)
#         b = cvxopt.matrix(np.zeros(A.shape[0]))
#         cvxopt.umfpack.solve(Ac, numeric, b)
#
# test_umfpack_lu_factorization_psd()
# test_umfpack_lu_factorization_indefinite()