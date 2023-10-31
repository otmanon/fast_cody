import numpy as np
import scipy.sparse as sp

'''
Generates a pyramid-like geometry for each rig bone

Inputs:
    P: b x 3 x 4 world transformation of each bone
    lengths: b x 1 length of each bone
    s: float, scale of the pyramid (default 0.1)
    return_selection_matrices: bool, if true, returns selection matrices mapping bone to rig geometry (default False)
Outputs:
    rV: 5b x 3 vertices of rig geometry
    rF: 7b x 3 faces of rig geometry
    SV: 5b x b selection matrix mapping rig geometry to bones
    SF: 7b x b selection matrix mapping rig geometry to faces
'''
def rig_geometry(P, lengths, s=0.1, return_selection_matrices=False):
    """
    Generates a pyramid-like geometry for each rig bone

    Parameters
    ----------
    P : (b, 3, 4) float numpy array
        World transformation of each bone
    lengths : (b, 1) float numpy array
        Length of each bone
    s : float, optional
        Scale of the pyramid
    return_selection_matrices : bool
        If True, returns selection matrices mapping bone to rig geometry (default False)

    Returns
    -------
    rV : (5b, 3) numpy float array
        Vertices of rig geometry
    rF : (7b, 3) numpy float array
        Faces of rig geometry.
    SV : (5b, b) scipy sparse matrix
        Selection matrix mapping rig geometry to bones. Only returns if return_selection_matrices is True
    SF : (7b, b) scipy sparse matrix
        Selection matrix mapping rig geometry to faces. Only returns if return_selection_matrices is True

    """
    if (P.ndim == 2):
        P = P[None, :,  :]
    #make standard tetrahedron with square base vertices, pointed upwards
    V = np.array(
        [[-0.5, 0, -0.5 ],
         [0.5,  0, -0.5  ],
         [0.5,  0, 0.5 ],
         [-0.5, 0, 0.5  ],
         [0,    1,   0  ]])



    F = np.array(
        [[0, 1, 2],
        [0, 2, 3],
        [0, 1,4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 3, 1]
         ]
    )


    #append ones to V to make homoegeneous

    V1 = np.hstack((V, np.ones((V.shape[0], 1))))

    k = P.shape[0]


    # empty list of face indices
    rF = np.empty((0, 3), dtype=int)
    # empty list of vertices
    rV = np.empty((0, 3), dtype=float)

    # initialize empty sparse matrices SV and SF
    SV = sp.coo_matrix((0, k))
    SF = sp.coo_matrix((0, k))

    for b in range(k):
        Tb = P[b, :, :]

        S = np.diag([s, lengths[b], s, 1])
        rVbs = (S@V1.T).T
        rVb = (Tb@rVbs.T).T

        # append face indices
        rF = np.vstack((rF, F + b*5))
        rV = np.vstack((rV, rVb))

        # selection matrix for vertices associated with each weights
        iV = np.arange(0, 5)
        jV = b*np.ones(5)
        vV = np.ones(5)
        SVb = sp.coo_matrix((vV, (iV, jV)), shape=(V.shape[0], k))

        iF = np.arange(0, 7)
        jF = b*np.ones(7)
        vF = np.ones(7)
        SFb = sp.coo_matrix((vF, (iF, jF)), shape=(F.shape[0], k))


        if return_selection_matrices:
            #concatenate SVb to global SV
            if b == 0:
                SV = SVb
                SF = SFb
            else:
                SV = sp.vstack((SV, SVb))
                SF = sp.vstack((SF, SFb))


    if return_selection_matrices:
        return rV, rF, SV, SF
    else:
        return rV, rF


