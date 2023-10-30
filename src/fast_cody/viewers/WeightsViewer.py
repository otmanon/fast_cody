import os

import numpy as np
import polyscope as ps

import time

from fast_cody.rig_geometry import rig_geometry

import polyscope.imgui as psim
class WeightsViewer():
    def __init__(self, V, F, W,
              eye_pos = None,
              eye_target = None,
              path="",
              R=np.identity(3), period=1,
                 vminmax=None, P=None, s=None, l=None, alpha=1):

        write_png = False
        if (path != ""):
            write_png = True
            os.makedirs(path, exist_ok=True)

        ps.init()

        self.path = path
        if F.shape[1] == 1:
            self.mesh = ps.register_point_cloud("mesh", V @ R.T, color=[1, 0.1, 0.1], transparency=alpha)
        elif F.shape[1] == 2:
            self.mesh = ps.register_curve_network("mesh", V @ R.T, F, color=[1, 0.1, 0.1], transparency=alpha)
        elif F.shape[1] == 3:
            self.mesh = ps.register_surface_mesh("mesh", V @ R.T, F, color=[1, 0.1, 0.1], transparency=alpha)
        elif F.shape[1] == 4:
            self.mesh = ps.register_volume_mesh("mesh", V @ R.T, F, color=[1, 0.1, 0.1], transparency=alpha)

        self.W = W
        self.mesh.add_scalar_quantity("weights", np.zeros((V.shape[0])),  enabled=True, cmap='coolwarm')
        self.i = 0
        self.write_png = write_png
        self.period = period
        self.vminmax = vminmax
        self.max_frame = self.W.shape[1]


        self.id = 0
        self.draw_bones = False
        if (P is not None):
            self.draw_bones = True
            self.P = P
            if s is None:
                s = 2
            if l is None:
                l = np.ones((W.shape[1]))
            self.s = s
            self.l = l
            [rV, rF, sV, sF] = rig_geometry(self.P[0, :, :], self.l, s=self.s)
            self.bone_mesh = ps.register_surface_mesh("bone_mesh", rV, rF)
            alpha = 0.5
            self.mesh.set_transparency(alpha)

        if eye_pos is not None and eye_target is not None:
            ps.look_at(eye_pos, eye_target)
        ps.set_user_callback(self.anim)
        ps.show()

        self.mesh.remove()
        if (self.draw_bones):
            self.bone_mesh.remove()


    def anim(self):
        if (self.i < self.max_frame):
            mesh = self.mesh
            i = self.i
            w = np.abs(self.W[:, i]).max()
            self.mesh.add_scalar_quantity("weights", self.W[:, i],  enabled=True, vminmax=[-w, w], cmap='coolwarm')
            # ps.set_camera_rotation(self.R)
            if (self.write_png):
                ps.screenshot(self.path + "/" + str(i).zfill(4) + ".png", False)
            if (self.draw_bones):
                [rV, rF, sV, sF] = rig_geometry(self.P[i, :, :], self.l, s=self.s)
                self.bone_mesh = ps.register_surface_mesh("bone_mesh", rV, rF)

            self.i += 1
            time.sleep(self.period)

        else:
            changed, ID = psim.SliderInt("bone ID", self.id, v_min=0, v_max=self.W.shape[1] - 1)
            if changed:
                self.id = ID

                w = np.abs(self.W[:, self.id]).max()
                self.mesh.add_scalar_quantity("weights", self.W[:, self.id], enabled=True,
                                              vminmax=[-w, w], cmap='coolwarm')
                if (self.draw_bones):
                    [rV, rF, sV, sF] = rig_geometry(self.P[self.id, :, :], self.l, s=self.s)
                    self.bone_mesh = ps.register_surface_mesh("bone_mesh", rV, rF)

        return