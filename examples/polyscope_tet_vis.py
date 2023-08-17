import polyscope as ps
import fast_cd_pyb as fcd
import numpy as np
import igl 
from sklearn.cluster import KMeans
import scipy.sparse as sp
name = "charizard"
mesh_file = "./data/" + name + ".msh"

cache_dir = "./cache/" + name + "/"
meta_file = cache_dir + "/meta.json"
[V, F, T] = fcd.readMSH(mesh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
F = igl.boundary_facets(T);

W = np.ones((V.shape[0], 1))
J = fcd.lbs_jacobian(V, W)

mode_type = "skinning"
num_modes = 10;
num_clusters = 100;
num_clustering_features = 10;
[B, Ws, L] = fcd.get_modes(V, T, W, J, mode_type, num_modes)
[l_clusters, C] = fcd.compute_clusters_weight_space(T, B, L, num_clusters, num_clustering_features)
   

index = 0
tet_var = 200
step = 0
def callback():
    global  mode, index, step
    if mode == 0:
        visible_tets = np.arange(0, min(index + tet_var, T.shape[0]-1) )
        mesh = ps.register_volume_mesh(name="Tet", vertices=V, tets=T[visible_tets, :])
        mesh.add_scalar_quantity(name="labels", values=l_tets[visible_tets], defined_on= 'cells', enabled=True, cmap='rainbow')


        ps.screenshot("./results/tet_cluster_vis/tets/" + str(step).zfill(4) + ".png", transparent_bg=False)

        index += tet_var
        step += 1
        if (index >= T.shape[0]):
            mode += 1
            index = 0;
    elif mode == 1:
        is_cluster_visible = np.zeros((num_clusters, 1))
        is_cluster_visible[0:index] = 1
        is_tet_visible = I.T @ is_cluster_visible
        visible_tets, _n = np.where(is_tet_visible)
        mesh = ps.register_volume_mesh(name="Tet", vertices=V, tets=T[visible_tets, :])
        mesh.add_scalar_quantity(name="labels", values=l_clusters[visible_tets], defined_on= 'cells', enabled=True, cmap='rainbow',vminmax=[0, num_clusters])
  
        ps.screenshot("./results/tet_cluster_vis/clusters/" + str(index).zfill(4) + ".png", transparent_bg=False)  
        index += 1
        if (index == num_clusters):
            mode += 1
            index = 0
    elif (mode == 2):
        quit(); 
    
    
ps.set_user_callback(callback);
ps.show()

