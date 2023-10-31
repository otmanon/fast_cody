
import numpy as np
def rotate_rig(P, R):
    """
    Rotates a rig by a rotation matrix R

    Parameters
    ----------
    P : (b, 3, 4) float numpy array
        World transformation of each bone
    R : (3, 3) float numpy array
        Rotation matrix

    Returns
    -------
    Prot : (b, 3, 4) float numpy array
        Rotated world transformation of each bone

    """
    k = P.shape[0]
    if (len(P.shape) == 3):
        k = P.shape[0]
        Prot = np.zeros(P.shape)
        for b in range(k):
            # stack a row of [0 0 0 1] tp T0
            # same with T
            T = P[b, :, :]
            Trot = R @ T
            Prot[b, :, :] = Trot[:3, :]

    if (len(P.shape) == 4 ):
        frames = P.shape[0]
        k = P.shape[1]
        Prot = np.zeros(P.shape)
        for frame in range(frames):
            for b in range(k):
                # stack a row of [0 0 0 1] tp T0
                T = P[frame, b, :, :]
                Trot = R @ T
                Prot[frame, b, :, :] = Trot[:3, :]

    return Prot

