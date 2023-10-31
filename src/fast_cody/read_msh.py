import fast_cd_pyb as fcdp


def read_msh(msh_file):
    """
    Reads .msh file generated by TetWild.

    Parameters
    ----------
    msh_file : str
        path to Tet mesh .msh file (usually generated by TetWild)

    Returns
    -------
    V : (n, 3) float numpy array
        vertex positions
    F : (f, 3) int numpy array
        triangle indices
    T : (m, 4) int numpy array
        tetrahedra indices

    """

    [V, F, T] = fcdp.readMSH(msh_file)
    return V, F, T
