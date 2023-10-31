import numpy as np
import json
'''
Reads a .json file containing rig information saved by the rigrats addon.

Input:
    rig_file: path to the .json file
Output:
    V: n x 3 vertices of mesh the rig is tied to
    F: f x 3 triangle faces of mesh the rig is tied to
    W: n x b skinning weights
    P0: b x 3 x 4  world transformation of each bone
    lengths: b x 1 length of each bone
    pI: b x 1 parent indices of each bone (-1 if root)
'''
def read_rig_from_json(rig_file):
    """  Reads a .json file containing rig information saved by the rigrats addon.

    Input:
        rig_file: path to the .json file
    Output:
        V: n x 3 vertices of mesh the rig is tied to
        F: f x 3 triangle faces of mesh the rig is tied to
        W: n x b skinning weights
        P0: b x 3 x 4  world transformation of each bone
        lengths: b x 1 length of each bone
        pI: b x 1 parent indices of each bone (-1 if root)

    """

    with open(rig_file) as f:
        rig = json.load(f)

    V = np.array(rig["V"])
    F = np.array(rig["F"])
    W = np.array(rig["W"])
    lengths = np.array(rig["lengths"])
    pI = np.array(rig["pI"])
    P0 = np.array(rig["p0"])

    return V, F, W, P0, lengths, pI