import scipy as sp
import numpy as np
import igl

from .deformation_jacobian import deformation_jacobian
from .vectorized_transpose import vectorized_transpose
from .vectorized_trace import vectorized_trace

'''
Computes the linear elasticity hessian matrix
'''
def linear_elasticity_hessian(V, T, mu=None, lam=None):
    """ Linear elasticity hessian matrix. The second derivative of the following energy
        ```
        E_{linear elasticity} = Σ mu e:e +  lam/2  (tr(e))^2
        ```
        Where e is the small strain tensor `e = 1/2(F^T +F)`
        https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf

        Parameters
        ----------
        V : (n, 3) numpy float array
            Rest vertex geometry
        T : (t, 4) numpy int array
            Tetrahedron indices
        mu : float or (t, 1) numpy float array or None
            First lame parameter. if None, then sets it to 1 for all tets.
        lam : float or (t, 1) numpy float array or None
            Second lame parameter. if None, then sets it to 0 for all tets.
    """
    dim = V.shape[1]
    B = deformation_jacobian(V, T);

    if (mu is None):
        mu = np.ones((T.shape[0]))
    elif (np.isscalar(mu)):
        mu = mu* np.ones((T.shape[0]))
    if (lam is None):
        lam = 0* np.ones((T.shape[0]))
    elif (np.isscalar(lam)):
        lam =  lam* np.ones((T.shape[0]))


    muv = np.repeat(mu, V.shape[1])
    muv = np.tile(muv, (V.shape[1]))
    Mu = sp.sparse.diags(muv)

    lamv = np.repeat(lam, V.shape[1])
    lamv = np.tile(lamv, (V.shape[1]))
    Lam = sp.sparse.diags(lamv)

    y = np.arange(0, Lam.shape[0])

    #m = np.arange(0, 12).reshape(6, 2)
    # mvec = m.flatten(order="F")
    # Tp = vectorized_transpose(3,2)
    Tp = vectorized_transpose(T.shape[0], d=dim)
    # Tr = vectorized_trace(3, d=dim)
    Tr = vectorized_trace(T.shape[0], d=dim)

    # yt = Tp @ y
    #
    # ytr = Tr @ y
    # yt.reshape(F.shape[0], 3, 3 )
    # vol = np.ones(F.shape[0])
    if (dim == 2):
        vol = igl.doublearea(V, T) / 2
    elif (dim == 3):
        vol = igl.volume(V, T)

    Vol = np.repeat(vol, V.shape[1])
    Vol = np.tile(Vol, (V.shape[1]))
    Vol = sp.sparse.diags(Vol) #Should just make a function for volume matrix

    dp2deps2 = 2* Mu + Lam @ Tr.T @  Tr

    I = sp.sparse.identity(Tp.shape[0])
    depsdf = (I + Tp)*0.5
    d2epsdf2 = depsdf.T @ dp2deps2 @ depsdf
    H = B.T @ Vol @ d2epsdf2 @ B
    return H