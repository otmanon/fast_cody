import igl
import numpy as np
import scipy as sp
import igl as igl

import fast_cd_pyb as fcd

class interactive_handle_viewer():
    def __init__(self, V, T, T0, guizmo_callback, pre_draw_callback,
                 texture_png=None, texture_obj=None, t0=None, s0=None):
        viewer = fcd.fast_cd_viewer()
        F = igl.boundary_facets(T)
        vis_texture = False
        if texture_png is not None and texture_obj is not None:
            vis_texture = True
        self.vis_texture = vis_texture

        if not vis_texture:
            viewer.set_mesh(V, F, 0)
            viewer.invert_normals(True, 0)
            color = np.array([144, 210, 236]) / 255.0
            viewer.set_color(color, 0)
            viewer.init_guizmo(True, T0, guizmo_callback, "translate")
            viewer.set_pre_draw_callback(pre_draw_callback)
            self.V = V
        if vis_texture:
            [Vf, TC, N, Ff, FTC, FN] = fcd.readOBJ_tex(texture_obj)
            if s0 is not None:
                Vf = Vf * s0
            if t0 is not None:
                Vf = Vf - t0

            P = sp.sparse.kron(sp.sparse.identity(3), fcd.prolongation(Vf, V, T))
            viewer.set_mesh(Vf, Ff, 0)
            viewer.set_show_lines(False, 0)
            viewer.set_texture(texture_png, TC, FTC, 0)
            viewer.set_face_based(False, 0)
            viewer.init_guizmo(True, T0, guizmo_callback, "translate")
            viewer.set_pre_draw_callback(pre_draw_callback)
            self.V = Vf
            self.P = P
        self.viewer = viewer

    def launch(self):
        self.viewer.launch()

    def update_displacement(self, U):
        if self.vis_texture:
            U = self.P @ U.reshape((-1, 1), order='F')
            U = U.reshape((-1, 3), order='F')
        X = U + self.V
        self.viewer.set_vertices(X, 0)
        self.viewer.compute_normals(0)


