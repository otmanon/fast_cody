import numpy as np
'''
From the global affine matrix of each bone, and each bone's parent index, make a bidirectional
edge simplex representing the rig

inputs
P0: b x 3 x 4 the world affine matrix of each bone
pI: b x 1 the parent index of each bone
outputs:
rV: Vx3 the vertex geometry of the rig edge simplex
rE: e x 2 the edge list of the rig edge simplex 
'''
def rig_curve_geometry(P0, pI):
    """
    From the global affine matrix of each bone, and each bone's parent index, make a bidirectional
    edge simplex representing the rig

    Parameters
    ----------
    P0 : (b, 3, 4) numpy float  array
        The world affine matrix of each bone
    pI : (b, 1)  numpy int array
        The parent index of each bone

    Returns
    -------
    rV : (V, 3) numpy float array
        The vertex geometry of the rig edge simplex
    rE : (e, 2) numpy int array
        The edge list of the rig edge simplex

    """

    assert(pI.shape[0] > 1)
    assert(P0.shape[0] == pI.shape[0])
    rV = P0[:, :,3]

    rE = np.zeros((0, 2), dtype=np.int32)
    # fill out E, the edge list
    for i in range(len(pI)):
        if (pI[i] == -1):
            continue
        else:
            # make a bidirectional edge between
           #i and pI[i]
            ei = np.array([[i, pI[i]]])
            # stack it on top of rE
            rE = np.vstack((rE, ei))

    # import polyscope as ps
    # ps.init()
    # ps.register_curve_network("rig", rV, rE)
    # ps.register_point_cloud("rV", rV)
    # ps.show()


    return rV, rE