import scipy as sp
import numpy as np

from .deformation_jacobian import deformation_jacobian
from .vectorized_transpose import vectorized_transpose
from .vectorized_trace import vectorized_trace

'''
Computes the linear elasticity hessian matrix
'''
def linear_elasticity_hessian(V, F, mu=None, lam=None):
    dim = V.shape[1]
    B = deformation_jacobian(V, F);

    if (mu is None):
        mu = np.ones((F.shape[0]))
    elif (np.isscalar(mu)):
        mu = mu* np.ones((F.shape[0]))
    if (lam is None):
        lam = 0* np.ones((F.shape[0]))
    elif (np.isscalar(lam)):
        lam =  lam* np.ones((F.shape[0]))


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
    Tp = vectorized_transpose(F.shape[0], d=dim)
    # Tr = vectorized_trace(3, d=dim)
    Tr = vectorized_trace(F.shape[0], d=dim)

    # yt = Tp @ y
    #
    # ytr = Tr @ y
    # yt.reshape(F.shape[0], 3, 3 )
    # vol = np.ones(F.shape[0])
    if (dim == 2):
        vol = gtb.doublearea(V, F)/2
    elif (dim == 3):
        vol = gtb.volume(V, F)

    Vol = np.repeat(vol, V.shape[1])
    Vol = np.tile(Vol, (V.shape[1]))
    Vol = sp.sparse.diags(Vol) #Should just make a function for volume matrix

    dp2deps2 = 2* Mu + Lam @ Tr.T @  Tr

    I = sp.sparse.identity(Tp.shape[0])
    depsdf = (I + Tp)*0.5
    d2epsdf2 = depsdf.T @ dp2deps2 @ depsdf
    H = B.T @ Vol @ d2epsdf2 @ B
    return H