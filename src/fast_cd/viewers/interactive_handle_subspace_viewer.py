import igl
import numpy as np
import scipy as sp
import igl as igl

import fast_cd_pyb as fcd

import fast_cd

'''
Viewer for interactive affine handle app. This viewer only updates the draw call by sending
JUST the subspace coefficients and rig parameters to the GPU, effectively doing the full space
projection required for rendering entirely in the vertex shader.

Inputs:
V - n x 3 mesh geometry
T - t x 4 tet indices
Wp - n x 1 primary skinning weights for rig
Ws - n x m secondary skinning weights for subspace
T0 - 4 x 4 initial rig transform in world space
guizmo_callback - callback function for guizmo widget
pre_draw_callback - callback function for pre-draw step

Optional:
texture_png - path to texture png file
texture_obj - path to texture obj file
t0 - initial mesh translation (to align the textured mesh with)
s0 - initial mesh scale (to align the textured mesh with)
'''
class interactive_handle_subspace_viewer():
    def __init__(self, V, T, Wp, Ws, T0, pre_draw_callback,
                 texture_png=None, texture_obj=None, t0=None, s0=None):
        vertex_shader_path = fast_cd.get_shader("./vertex_shader_16.glsl")
        fragment_shader_path = fast_cd.get_shader("./fragment_shader.glsl")

        viewer = fcd.fast_cd_viewer_custom_shader(vertex_shader_path,
                                                  fragment_shader_path, 16, 16)
        print("  g        Toggle guizmo widget")
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

        self.T0 = T0
        self.transform = "translate"
        viewer.init_guizmo(True, T0, self.guizmo_callback, self.transform)

        viewer.set_pre_draw_callback(pre_draw_callback)

        viewer.set_key_callback(self.callback_key_pressed)

        self.viewer = viewer

    def guizmo_callback(self, A):
        self.T0 = A

    def callback_key_pressed(s, key, modifier):
        if (key == ord('g')):
            if (s.transform == "translate"):
                s.transform = "rotate"
            elif (s.transform == "rotate"):
                s.transform = "scale"
            elif (s.transform == "scale"):
                s.transform = "translate"

            s.viewer.change_guizmo_op(s.transform)
        return False
    def launch(self):
        self.viewer.launch(60, True)

    def change_guizmo_op(self, op):
        self.viewer.change_guizmo_op(op)
    def update_subspace_coefficients(self, z, p):
        self.viewer.set_bone_transforms(p, z, 0);
        self.viewer.updateGL(0)


