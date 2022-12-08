import scipy as sp

def build_vertex_rotation_matrix(R, I, n, dim): 
    """
    Builds a rotation matrix for each vertex in the mesh
    :param R: rotation matrix
    :param I: vertex indices
    :param n: number of vertices
    :param dim: dimension of the mesh
    :return: sparse matrix Rv that when multiplied against flattend v, gives us rotation matrix
    Assumes vertices are 3D
    """
    Rv = sp.sparse.csc_matrix((3 * n, 3 *n))
    for ii in range(0, I.shape[0]):
        i = [I[ii], I[ii] + n, I[ii] + 2 * n,I[ii], I[ii] + n, I[ii] + 2 * n, I[ii], I[ii] + n, I[ii] + 2 * n ];
        j = [ I[ii],  I[ii],I[ii], I[ii] + n,  I[ii] + n,  I[ii] + n, I[ii] + 2*n,  I[ii] + 2*n,  I[ii] + 2*n] ;
        v = [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2]];
        Rv = Rv + sp.sparse.csc_matrix((v, (i, j)), shape=(3*n, 3*n))
    return Rv
