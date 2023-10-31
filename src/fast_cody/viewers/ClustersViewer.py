import os

import numpy as np
import polyscope as ps

import time


import polyscope.imgui as psim
class ClustersViewer():
    """
    This class is used to visualize the clusters of a mesh using polyscope.

    Parameters
    ----------
    V : float numpy array
        n x 3 vertex positions
    T : int numpy array
        m x 4 tetrahedra indices
    l : int numpy array
        T x 1  per-tet cluster indices.
    eye_pos : float numpy array
        3 x 1 position of the camera
    eye_target : float numpy array
        3 x 1 position of the camera target
    path : str
        path to save the images to
    R : float numpy array
        3 x 3 rotation matrix to apply to the mesh
    period : float
        time between frames
    vminmax : float numpy array
        2 x 1 min and max values for the color map
    alpha : float
        transparency of the mesh
    grouped : bool
        whether to group the clusters or not

    Examples
    --------
    >>> import fast_cody as fcd
    >>> [V, F, T] = fcd.read_msh(fcd.get_data("cd_fish.msh"))
    >>> [B, l, Ws] = fcd.skinning_subspace(V, T, 10, 100)
    >>> fcd.viewers.ClustersViewer(V, T, l)
    """
    def __init__(self, V, T, l,
              eye_pos = [2, 2, 2],
              eye_target = [0,0, 0],
              path="",
              R=np.identity(3), period=1/60,
                 vminmax=None, alpha=1, grouped=True):
        write_png = False
        if (path != ""):
            write_png = True
            os.makedirs(path, exist_ok=True)

        nc = np.max(l)+1
        ps.init()
        ps.look_at(eye_pos, eye_target)

        self.grouped = grouped
        self.V = V
        self.T = T
        self.path = path
        self.R = R
        self.l = l
        self.create_mesh(V@ R.T, T, l)
        self.i = 0
        self.write_png = write_png
        self.period = period
        self.vminmax = vminmax
        self.max_frame = nc
        self.id = 0
        self.nc = nc
        arr = np.arange(0, nc)
        np.random.shuffle(arr)
        self.arr = arr
        ps.set_user_callback(self.anim)
        ps.show()
        self.mesh.remove()


    def anim(self):
        if not (self.grouped):
            if (self.i < self.max_frame):
                mesh = self.mesh
                i = self.i
                ind = self.l == self.arr[i]
                self.create_mesh(self.V @ self.R.T, self.T[ind, :], self.l[ind])
                # ps.set_camera_rotation(self.R)
                if (self.write_png):
                    ps.screenshot(self.path + "/" + str(i).zfill(4) + ".png", False)
                self.i += 1
                time.sleep(self.period)
            else:
                changed, ID = psim.SliderInt("cluster label", self.id, v_min=0, v_max=self.nc)
                if changed:
                    self.id = ID
                    ind = self.l == ID
                    self.create_mesh(self.V @ self.R.T, self.T[ind, :], self.l[ind])
        else:
            if (self.i < self.max_frame):
                mesh = self.mesh
                i = self.i
                ind= np.zeros(self.T.shape[0], dtype=bool)
                for j in range(0, i+1):
                    ind = np.logical_or(ind,  (self.l == self.arr[j]))

                self.create_mesh(self.V @ self.R.T,  self.T[ind, :], self.l[ind])

                # ps.set_camera_rotation(self.R)
                if (self.write_png):
                    ps.screenshot(self.path + "/" + str(i).zfill(4) + ".png", False)
                self.i += 1
                time.sleep(self.period)
            else:
                self.create_mesh(self.V @ self.R.T, self.T, self.l)

        return



    def create_mesh(self, X, T, l):
        nc = np.max(l)

        if T.shape[1] == 4:
            self.mesh = ps.register_volume_mesh("mesh", X , T)
            self.mesh.add_scalar_quantity("clusters", l, defined_on='cells', cmap='rainbow', enabled=True, vminmax=[0, nc])
        elif T.shape[1] == 2:
            self.mesh = ps.register_curve_network("mesh", X, T)
            self.mesh.add_scalar_quantity("clusters", l, defined_on='edges', cmap='rainbow', enabled=True,
                                          vminmax=[0, nc])
