import os

import numpy as np
import scipy as sp
import igl

import json

import fast_cd_pyb as fcd
import fast_cd as fc
from os.path import basename, splitext


def interactive_cd_affine_handle(mesh_file, mu=1e4, rho=1e3, num_modes=16, num_clusters=100,
                                 results_dir=None, read_cache=False, texture_png=None, texture_obj=None):
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

    Wp = np.ones((V.shape[0], 1)) #single handle skinning weight
    J = fc.lbs_jacobian(V, Wp)
    modes_dir =  cache_dir
    clusters_dir = cache_dir

    C = fc.complementary_constraint_matrix(V, T, J, dt=1e-3)
    C2 = fc.lbs_weight_space_constraint(V, C)
    [B, l, Ws] = fc.skinning_subspace(V, T, num_modes, num_clusters, C=C2, read_cache=read_cache,
                                     cache_dir=cache_dir, constraint_enforcement="project");
    # fc.WeightsViewer(V, T, W)
    # fc.ClustersViewer(V, T, l)
    sim = fc.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=1e-2, cache_dir=cache_dir, read_cache=read_cache)

    # set sim state and initial rig parameters
    z0 = np.zeros((num_modes*12, 1))
    T0 = np.identity(4).astype( dtype=np.float32, order="F");
    p0 = T0[0:3, :].reshape((12, 1))
    st = fc.fast_cd_state(z0, p0)

    step = 0
    def pre_draw_callback():
         nonlocal J, B, T0, sim, st, step
         p = T0[0:3, :].reshape( (12, 1))

         z = sim.step( p, st)

         st.update(z, p)
         # print(np.linalg.norm(B@z));

         U = np.reshape(J @ p + B @ z, (J.shape[0]//3, 3), order="F")

         viewer1.update_subspace_coefficients(z, p)
         # viewer2.update_displacement(U - V)

         step += 1

    def guizmo_callback(A):
        nonlocal T0
        T0 = A

    viewer1 = fc.viewers.interactive_handle_subspace_viewer(V, T, Wp, Ws, T0, guizmo_callback, pre_draw_callback,
                                                  texture_png=texture_png, texture_obj=texture_obj,
                                                  t0=to, s0=so )

    # viewer2 = fc.viewers.interactive_handle_viewer(V, T, T0, guizmo_callback, pre_draw_callback,
    #                                               texture_png=texture_png, texture_obj=texture_obj,
    #                                               t0=to, s0=so)
    viewer1.launch()