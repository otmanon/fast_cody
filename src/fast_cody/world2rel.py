
import numpy as np
def world2rel(P, P0):
    """
    Converts world coordinates to relative coordinates
    Parameters
    ----------
    P : (frames, b, 3, 4) float numpy array
        World transformation of each bone
    P0 : (b, 3, 4) float numpy array
        World transformation of each bone at the first frame

    Returns
    -------
    Prel : (frames, b, 3, 4) float numpy array
        Relative transformation of each bone
    """
    frames = P.shape[0]
    k = P0.shape[0]


    Prel = np.zeros(P.shape)
    for frame in range(frames):
        for b in range(k):
            # stack a row of [0 0 0 1] tp T0
            T0 = np.vstack((P0[b, :, :], np.array([0, 0, 0, 1])))

            # same with T
            T = np.vstack((P[frame,b, :, :], np.array([0, 0, 0, 1])))
            Trel = T @ np.linalg.inv(T0)
            Prel[frame, b, :, :] = Trel[:3, :]

    return Prel

