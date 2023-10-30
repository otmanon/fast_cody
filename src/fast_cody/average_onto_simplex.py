
import numpy as np
''' 
Averages rows of A onto the simplex defined by T

Inputs:
    A - n x d  per vertex d-dimensinoal quantities
    T - t x s  simplex indices

Returns:
    At - t x d  per simplex averaged d-dimensional quantities
'''
def average_onto_simplex(A, T):
    """ Average quantity from vertices to simplices
    Parameters
    ----------
    A : (n, d) numpy float array
        Per vertex d-dimensional quantities
    T : (t, s) numpy int array
        Simplex indices

    Returns
    -------
    At : (t, d) numpy float array
        Per simplex d-dimensional quantities
    """
    At = np.zeros((T.shape[0], A.shape[1]))
    for td in range(T.shape[1]):
        At += (A[T[:, td], :])/T.shape[1]

    return At