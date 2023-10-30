import os

import numpy as np
import polyscope as ps

import time

from fast_cody.rig_geometry import rig_geometry

import polyscope.imgui as psim
class ClustersViewer():
    def __init__(self, V, T, l,
              eye_pos = [0, 0, 0],
              eye_target = [0, 5, 5],
              path="",
              R=np.identity(3), period=1/60,
                 vminmax=None, alpha=1, grouped=True):
        write_png = False
        if (path != ""):
            write_png = True
            os.makedirs(path, exist_ok=True)

        nc = np.max(l)+1
        ps.init()
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
