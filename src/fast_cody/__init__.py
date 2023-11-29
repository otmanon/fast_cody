__version__ = '0.0.8'

from .arap_hessian import arap_hessian
from .skinning_subspace import  skinning_subspace
from .average_onto_simplex import  average_onto_simplex
from .closest_orthogonal_subspace import  closest_orthogonal_subspace
from .cluster_centroids import cluster_centroids_spectral, cluster_centroids_euclidean
from .cluster_grouping_matrices import  cluster_grouping_matrices
from .deformation_jacobian import  deformation_jacobian
from .laplacian import laplacian
from .laplacian_eigenmodes import laplacian_eigenmodes
from .lbs_jacobian import lbs_jacobian
from .lbs_weight_space_constraint import lbs_weight_space_constraint
from .linear_elasticity_hessian import linear_elasticity_hessian
from .normalize_height_and_center import normalize_height_and_center
from .orthonormalize import orthonormalize
from .project_into_subspace import project_into_subspace
from .read_rig_from_json import read_rig_from_json
from .read_rig_anim_from_json import read_rig_anim_from_json
from .rig_curve_geometry import rig_curve_geometry
from .rig_geometry import rig_geometry
from .rotate_rig import rotate_rig
from .skinning_clusters import skinning_clusters
from .vectorized_trace import vectorized_trace
from .vectorized_transpose import vectorized_transpose
from .ympr_to_lame import ympr_to_lame
from .project_out_subspace import project_out_subspace
from .diffuse_weights import diffuse_weights
from .momentum_leaking_matrix import momentum_leaking_matrix
from .complementary_constraint_matrix import complementary_constraint_matrix
from .umfpack_lu_solve import umfpack_lu_solve
from .eigs import eigs
from .fast_cd_sim import fast_cd_sim, fast_cd_state
from .one_euro_filter import OneEuroFilter
from .mediapipe_face_captor import mediapipe_face_captor
from .world2rel import world2rel
from .read_msh import read_msh

#Apps
from .apps.interactive_cd_rig_anim import interactive_cd_rig_anim
from .apps.interactive_cd_face_tracking import interactive_cd_face_tracking
from .apps.interactive_cd_affine_handle import interactive_cd_affine_handle

#Viewers
from .viewers.WeightsViewer import WeightsViewer
from .viewers.ClustersViewer import ClustersViewer
from .viewers.interactive_handle_subspace_viewer import interactive_handle_subspace_viewer
# from .viewers.interactive_handle_viewer import interactive_handle_viewer

# set data path and shaders path
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_shader(path):
    return os.path.join(_ROOT, 'shaders', path)
def get_data(path):
    return os.path.join(_ROOT, 'data', path)
