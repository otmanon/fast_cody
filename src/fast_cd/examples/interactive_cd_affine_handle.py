import os

import numpy as np
import scipy as sp
import igl
import fast_cd_pyb as fcd
import fast_cd as fc
import json

from os.path import basename, splitext


def interactive_cd_affine_handle(mesh_file, mu=1, num_modes=16, num_clusters=100, results_dir=None, read_cache=False):
    write_cache= True

    ## parameters

    h = 1e-2
    do_inertia = True

    assert(splitext(mesh_file)[1] == '.msh' and "only supports .msh file format")
    name = basename(splitext(mesh_file)[0])
    mesh_file = mesh_file
    if results_dir is None:
        results_dir = "./results/interactive_cd_affine_handle/" + name + "/"
    cache_dir = results_dir + "/cache/"
    os.makedirs(cache_dir, exist_ok=True)

    [V, F, T] = fcd.readMSH(mesh_file)
    [V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]])) #center to unit height and about origin
    F = igl.boundary_facets(T);

    W = np.ones((V.shape[0], 1)) #single handle skinning weight
    J = fc.lbs_jacobian(V, W)

    modes_dir =  cache_dir
    clusters_dir = cache_dir

    C = fc.complementary_constraint_matrix(V, T, J, dt=1e-3)

    C2 = fc.lbs_weight_space_constraint(V, C)

    [B, l, W] = fc.skinning_subspace(V, T, num_modes, num_clusters, C=C2, read_cache=read_cache,
                                     cache_dir=cache_dir, constraint_enforcement="project");

    # B = fc.project_out_subspace(B, C.T)
    # W = igl.read_dmat(cache_dir + "/cpp/W.DMAT")
    # B = fc.lbs_jacobian(V, W)
    # l = igl.read_dmat(cache_dir + "/cpp/labels.DMAT")
    # fc.WeightsViewer(V, T, W)

    #
    # fc.ClustersViewer(V, T, l)

    solver_params = fcd.local_global_solver_params(True, 10, 1e-4)
    Aeq = sp.sparse.csc_matrix((0, 0))
    sim_params = fcd.fast_cd_arap_sim_params(V, T, B, l, sp.sparse.csc_matrix(J), Aeq, mu, h, do_inertia)
    sim = fcd.fast_cd_arap_sim(cache_dir, sim_params, solver_params, read_cache, write_cache)

    viewer = fcd.fast_cd_viewer()

    viewer.set_mesh(V, F, 0)
    viewer.invert_normals(True, 0)

    color = np.array([144, 210, 236])/255.0
    viewer.set_color(color, 0)

    # set sim state
    z0 = np.zeros((num_modes*12, 1))
    T0 = np.identity(4).astype( dtype=np.float32, order="F");
    p0 = T0[0:3, :].reshape( (12, 1))
    st = fcd.cd_sim_state(z0, z0, p0, p0)

    # momentum leaking matrix
    f_ext = np.zeros((z0.shape[0], 1))
    bc = np.array([[]], dtype=np.float64).T

    step = 0
    def callback():
         nonlocal J, B, T0, sim, st, step, f_ext
         p = T0[0:3, :].reshape( (12, 1))

         if step < 10:
             f_ext = 0*f_ext + 0.005 * np.random.rand(f_ext.shape[0], 1)
         else:
             f_ext = 0*f_ext
         #initial guess...
         z = st.z_curr
         z = sim.step(z, p,  st, f_ext, bc).reshape((12*num_modes, 1), order="F")


         st.update(z, p)
         # print(np.linalg.norm(B@z));

         U = np.reshape(J@p + sim.params().B@z, (int(J.shape[0]/3), 3), order="F")
         viewer.set_vertices(U, 0)
         viewer.compute_normals(0)
         step += 1



    def guizmo_callback(A):
        nonlocal T0
        T0 = A

    viewer.init_guizmo(True, T0, guizmo_callback, "translate")
    viewer.set_pre_draw_callback(callback)
    viewer.launch()