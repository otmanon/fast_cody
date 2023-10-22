import igl
import numpy as np
import scipy as sp
import igl as igl

import fast_cd_pyb as fcd

import fast_cd


class interactive_handle_subspace_viewer():
    def __init__(self, V, T, Wp, Ws, T0, guizmo_callback, pre_draw_callback,
                 texture_png=None, texture_obj=None, t0=None, s0=None,
                 callback_key_pressed=None):
        vertex_shader_path = fast_cd.get_shader("./vertex_shader_16.glsl")
        fragment_shader_path = fast_cd.get_shader("./fragment_shader.glsl")

        viewer = fcd.fast_cd_viewer_custom_shader(vertex_shader_path,
                                                  fragment_shader_path, 16, 16)
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
            viewer.set_pre_draw_callback(pre_draw_callback)
            self.V = V
            self.Wp =  Wp;  # primary rig weights
            self.Ws =  Ws;  # secondary subspace weights
            viewer.set_weights(self.Wp, self.Ws, 0)

        if vis_texture:
            [Vf, TC, N, Ff, FTC, FN] = fcd.readOBJ_tex(texture_obj)
            if s0 is not None:
                Vf = Vf * s0
            if t0 is not None:
                Vf = Vf - t0

            P = fcd.prolongation(Vf, V, T)
            Pe = sp.sparse.kron(sp.sparse.identity(3), P )
            viewer.set_mesh(Vf, Ff, 0)
            viewer.set_texture(texture_png, TC, FTC, 0)
            self.Wp_tex = P @ Wp;  # primary rig weights
            self.Ws_tex = P @ Ws;  # secondary subspace weights
            viewer.set_weights(self.Wp_tex, self.Ws_tex, 0)

            viewer.set_show_lines(False, 0)
            viewer.set_face_based(False, 0)
            self.V = Vf
            self.Pe = Pe

        transform = "translate"
        viewer.init_guizmo(True, T0, guizmo_callback, transform)

        viewer.set_pre_draw_callback(pre_draw_callback)
        if callback_key_pressed is not None:
            viewer.set_key_callback(callback_key_pressed)

        self.viewer = viewer



    def launch(self):
        self.viewer.launch(60, True)

    def change_guizmo_op(self, op):
        self.viewer.change_guizmo_op(op)
    def update_subspace_coefficients(self, z, p):
        self.viewer.set_bone_transforms(p, z, 0);
        self.viewer.updateGL(0)


