import fast_cd_pyb as fcd
import igl

mesh = "./data/bulldog/bulldog.msh"

[V, F, T] = fcd.readMSH(mesh)
F = igl.boundary_facets(T)
viewer = fcd.fast_cd_viewer()

viewer.set_mesh(V, F, 0)

viewer.launch()