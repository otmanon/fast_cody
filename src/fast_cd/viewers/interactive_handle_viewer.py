import numpy as np

import fast_cd_pyb as fcd

class interactive_handle_viewer():
    def __init__(self, V, F, T0, guizmo_callback, pre_draw_callback ):
        viewer = fcd.fast_cd_viewer()
        viewer.set_mesh(V, F, 0)
        viewer.invert_normals(True, 0)
        color = np.array([144, 210, 236]) / 255.0
        viewer.set_color(color, 0)


        viewer.init_guizmo(True, T0, guizmo_callback, "translate")
        viewer.set_pre_draw_callback(pre_draw_callback)

        self.viewer = viewer

    def launch(self):
        self.viewer.launch()
    def update_vertices(self, V):
        self.viewer.set_vertices(V, 0)
        self.viewer.compute_normals(0)

