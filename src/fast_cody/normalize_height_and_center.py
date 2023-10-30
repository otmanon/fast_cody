
def normalize_height_and_center(V, h, c, return_scale_trans=False):

    height = V[:, 1].max() - V[:, 1].min()

    s = h / height

    V = V * s

    d =  V.mean(axis=0) - c

    V -= d

    if return_scale_trans:
        return V, s, d
    return V

