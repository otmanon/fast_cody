
def normalize_height_and_center(V, h, c, return_scale_trans=False):
    """
    Normalize the height of the mesh to h and center it at c.

    Parameters
    ----------
    V : numpy float array
        Mesh vertices
    h : float
        Desired height
    c : numpy float array
        Desired center
    return_scale_trans : bool
        If True, return the scale and translation applied to V

    Returns
    -------
    V : numpy float array
        Rescaled/translated mesh vertices
    s : float
        Scale applied to V. only returned if return_scale_trans is True
    t : numpy float array
        translation applied to V. only returned if return_scale_trans is True
    """

    height = V[:, 1].max() - V[:, 1].min()

    s = h / height

    V = V * s

    d =  V.mean(axis=0) - c

    V -= d

    if return_scale_trans:
        return V, s, d
    return V

