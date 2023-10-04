
import numpy as np
''' 
Averages rows of A onto the simplex defined by T
'''
def average_onto_simplex(A, T):
    At = np.zeros((T.shape[0], A.shape[1]))
    for td in range(T.shape[1]):
        At += (A[T[:, td], :])/T.shape[1]

    return At