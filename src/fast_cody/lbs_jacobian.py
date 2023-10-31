import numpy as np

def lbs_jacobian(V, W):
    """ Linear Blend Skinning Jacobian

        Parameters
        ----------
        V : (n, d) numpy float array
            Mesh vertices
        W : (n, k) numpy float array
            Mesh skinning weights

        Returns
        -------
        J : (nd, d(d+1)k) numpy float array
            Linear blend skinning Jacobian matrix

    """
    n = V.shape[0]
    d = V.shape[1]
    k = W.shape[1]

    one_d1 = np.ones((d+1, 1))
    one_k = np.ones((k, 1))

    # append 1s to V to make V1 , homogeneous
    V1 = np.hstack((V, np.ones((V.shape[0], 1))))


    Wexp = np.kron( W, one_d1.T)
    V1exp = np.kron( one_k.T, V1)
    J = Wexp * V1exp
    Jexp = np.kron(np.identity(d), J)
    return Jexp

#
# import torch
#
# '''
# Linear Blend Skinning Jacobian
# Input:
#     V: vertices of the mesh (n x d)
#     W: weights of the mesh (n x k)
# Output:
#     J: (nd x d(d+1)k linear blend skinning Jacobian matrix
# '''
# def lbs_jacobian_torch(V, W):
#
#     # make sure V andc W are on same device
#     assert(V.device == W.device)
#     device = V.device
#
#     n = V.shape[0]
#     d = V.shape[1]
#     k = W.shape[1]
#
#     one_d1 = torch.ones((d+1, 1)).to(device)
#     one_k = torch.ones((k, 1)).to(device)
#
#     # append 1s to V to make V1 , homogeneous
#     V1 = torch.hstack((V, torch.ones((V.shape[0], 1)).to(device)))
#
#
#     Wexp = torch.kron( W, one_d1.T)
#     V1exp = torch.kron( one_k.T, V1)
#     J = Wexp * V1exp
#     Jexp = torch.kron(torch.eye(d).to(device), J)
#
#
#     return Jexp
#
