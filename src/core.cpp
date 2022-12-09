#include "fast_cd_arap_sim.h"
#include "rig_parameters.h"
#include "get_skeleton_mesh.h"
#include "skinning_modes.h"
#include "get_modes.h"
#include "compute_clusters_igl.h"
#include "fast_cd_viewer.h"
#include "lbs_jacobian.h"
#include "scale_and_center_geometry.h"
#include "read_fast_cd_sim_static_precompute.h"
#include "write_fast_cd_sim_static_precomputation.h"
#include "surface_to_volume_weights.h"
#include "fit_rig_to_mesh.h"
#include "momentum_leaking_matrix.h"
#include "read_rig_from_json.h"
#include "fast_cd_subspace.h"
#include "fast_cd_external_force.h"
#include "read_rig_anim_from_json.h"
#include "rig_parameters.h"

#include <igl/readMSH.h>
#include <igl/massmatrix.h>
#include <igl/readOBJ.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;
using namespace std;
PYBIND11_MODULE(fast_cd_pyb, m) {

    py::class_<fast_cd_subspace>(m, "fast_cd_subspace", "Helper class that \
        builds/reads/writes the subspaces necessary to run and test Fast CD\n",
        py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<int, string, string, int, int, bool, bool, string>(), "\
            	Initializes and configures subspace... does NOT build it yet.\n \
                Inputs: \n \
                num_modes - (int) number of modes in subspace\n \
                subspace_constraint_type - (string) either \"none\", or \"cd\" or \
                                            \"cd_momentum_leak\"\n \
                mode_type - (string) either \"skinning\" or \"displacement\"\n \
                num_clusters - (int) number of clusters\n \
                num_clustering_features - (int) number of clustering features used\n \
                split_components - (bool) whether to split the components of the clusters \n \
                \n  \
                Optional:\n\
                debug - (bool) whether to save and store debug info\n \
                output_dir - (string) output directory where we will write debug info\n\
            ", py::arg("num_modes"), py::arg("subspace_constraint_type"),
            py::arg("mode_type"), py::arg("num_clusters"),
            py::arg("num_clustering_features"),
            py::arg("split_components"),
            py::arg("debug") = false, py::arg("output_dir") = "")
        .def("init", &fast_cd_subspace::init, "\
            Computes modes + clusters from scratch \n \
            Inputs : \n \
                V -> | n | x3 geometry \n \
                T -> | T | x4 tet indices \n \
                J -> | c | x3n null space / linear orthogonality constraint")
        .def("init_with_cache", &fast_cd_subspace::init_with_cache, "\
            Computes modes + clusters from scratch, \
            with control over when to read / write from cache \n \
            Inputs:  \n \
                V -> | n | x3 geometry  \n \
                T -> | T | x4 tet indices \n \
                J -> 3n x | c | rig jacobian (not weighed by mass matrix or nothing) \n \
                read_cache->whether or not to \
                            attempt to read modes and clusters from cache. \n \
                write_cache->whether or not to write modes and clusters to cache. \n \
                modes_cache_dir->directory where mode cache is \n \
            (Optional) \n \
                clusters_cache_dir->directory where clusters cache is \n \
                recompute_modes_if_not_found->whether or \
                                            not to recompute modes from scratch \
                                            if not found in cache(default true) \n \
                recompute_clusters_if_not_found->whether or not to  \
                                                recompute clusters from scratich if \
                                                not found in cache(default true) \n \
            ")
        .def("write_to_cache", &fast_cd_subspace::write_to_cache, " \
                Writes modes and clusters to cache directories \n \
                modes_dir -> (string)where to save modes directory \
                             (both B.DMAT or W.DMAT and L.DMAT, modes + frequencies) \n \
                clusters_dir -> (string)where to save clusters directory(labels.DMAT)\
                ")

        .def("read_from_cache", &fast_cd_subspace::read_from_cache, "  \
                Reads modes and clusters from cache directories \n \
                Inputs: \n \
                    modes_dir - (string)directory where B.DMAT / W.DMAT \n \
                                    and L.DMAT is stored \
                        clusters_dir - (string) directory where \
                clusters labels.DMAT is stored \n \
                ")
        .def("read_clusters_from_cache", &fast_cd_subspace::read_clusters_from_cache,
            "  \
                Read clusters from cache directories \n  \
                Inputs: n \
                    clusters_dir - (string)directory where cluster labels.DMAT is stored \n \
                        ")
        .def("read_modes_from_cache", &fast_cd_subspace::read_modes_from_cache,
            "  \
                    Read clusters from cache directories \n \
                    Inputs: \n \
                        modes_dur - (string)directory where \
                                        cluster B.DMAT / W.DMAT and L.DMAT is stored\
                     ")
        .def("getB", [](fast_cd_subspace& sub) {return sub.B; })
        .def("getW", [](fast_cd_subspace& sub) {return sub.W; })
        .def("getL", [](fast_cd_subspace& sub) {return sub.L; })
        .def("getLabels", [](fast_cd_subspace& sub) {return sub.l; })
        .def("setB", [](fast_cd_subspace& sub, EigenDRef<MatrixXd> B) {sub.B = B; })
        .def("setW", [](fast_cd_subspace& sub, EigenDRef<MatrixXd> W) {sub.W = W; })
        .def("setL", [](fast_cd_subspace& sub, VectorXd& L) {sub.L = L; })
        .def("setlabels", [](fast_cd_subspace& sub, VectorXi& l) {sub.l = l; })
        .def_readwrite("B", &fast_cd_subspace::B, py::return_value_policy::reference_internal)
        .def_readwrite("W", &fast_cd_subspace::W, py::return_value_policy::reference_internal)
        .def_readwrite("L", &fast_cd_subspace::L, py::return_value_policy::reference_internal)
        .def_readwrite("labels", &fast_cd_subspace::l, py::return_value_policy::reference_internal)
        ;


       
    py::class_<fast_cd_arap_local_global_solver>(m, "fast_cd_arap_local_global_solver")
        .def(py::init<>())
        .def(py::init<EigenDRef<MatrixXd> , 
            EigenDRef<MatrixXd>, cd_arap_local_global_solver_params&>(), " \n \
            Constructs arap local global solver object used to solve \n \
            dynamics quickly for fast Complementary DYnamoics \n \
            Inputs : \n \
                 A - m x m system matrix \n \
                Aeq - c x m constraint rows that enforece Aeq z = b \n \
                as linear equality constraints \n \
                p - cd_arap_local_global_solver_params \n \
            ")
        .def(py::init<EigenDRef<MatrixXd>,
            EigenDRef<MatrixXd>, bool, int, double>(), " \n \
            Constructs arap local global solver object used to solve \n \
            dynamics quickly for fast Complementary DYnamoics \n \
            Inputs : \n \
                 A - m x m system matrix \n \
                Aeq - c x m constraint rows that enforece Aeq z = b \n \
                as linear equality constraints \n \
               run_solver_to_convergence - (bool)  \n \
                max_iters - (int) \n \
                convergence_threshold - (double) \n \
                            where to stop if || res || 2 drops below this \n \
            ")
        .def_readwrite("prev_solve_iters", &fast_cd_arap_local_global_solver::prev_solve_iters)
        .def_readwrite("prev_res", &fast_cd_arap_local_global_solver::prev_res);
        


    py::class_<fast_cd_arap_sim>(m, "fast_cd_arap_sim")
        .def(py::init<std::string&, fast_cd_sim_params&, 
            cd_arap_local_global_solver_params&, bool, bool>())
        .def(py::init<fast_cd_sim_params&, cd_arap_local_global_solver_params&>(), "\
            cache_dir - (string)  \n \
            sim_params - fast_cd_sim_params  \n \
            solver_params - solver_params \n  \
            read_cache - (bool) \n \
            write_cache - (bool)\
            ")
        .def("step", static_cast<VectorXd(fast_cd_arap_sim::*)(
            const VectorXd&, const VectorXd&, const cd_sim_state&,
            const  VectorXd&, const  VectorXd&)>
            (&fast_cd_arap_sim::step), " \ \
	        Advances the pre-configured simulation one step  \n \
            Inputs : \n \
                z:  m x 1 current guess for z(maybe shouldn't expose this) \n \
                p : p x 1 flattened rig parameters following writeup column order flattening converntion \n \
                state : sim_cd_state that contains info like z_curr, z_prev, p_currand p_prev \n \
                f_ext : used to specify excternal forces like gravity. \n \
                bc : rhs of equality constraint if some are configured in system \n \
                (should match in rows with sim.params.Aeq) \n \
            Outputs : \n \
                z_next: m x 1 next timestep degrees of freedom \n \
            ")
        .def("step", static_cast<VectorXd(fast_cd_arap_sim::*)(
            const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&,
            const VectorXd&, const VectorXd&,
            const  VectorXd&, const  VectorXd&)>
            (&fast_cd_arap_sim::step) , " \n \
            Advances the pre - configured simulation one step \n \
            Inputs : \n \
                z:  m x 1 current guess for z(maybe shouldn't expose this) \n \
                p : p x 1 flattened rig parameters following writeup column order flattening converntion  \n \
                z_curr : m x 1 current d.o.f.s  \n \
                z_prev : m x 1 previos d.o.f.s  \n \
                p_curr : p x 1 current rig parameters  \n \
                p_prev : p x 1 previous rig parameters  \n \
                f_ext : used to specify excternal forces like gravity.  \n \
                bc : rhs of equality constraint if some are configured in system  \n \
                (should match in rows with sim.params.Aeq)  \n \
            Outputs :  \n \
                z_next: m x 1 next timestep degrees of freedom  \n \
            ")
        .def("params", [](fast_cd_arap_sim& sim) {
            fast_cd_sim_params* p = (fast_cd_sim_params*)sim.params;
            return p;
        })
        .def("sp", [](fast_cd_arap_sim& sim) {
            fast_cd_arap_static_precomp* p = (fast_cd_arap_static_precomp*)sim.sp;
        return p;
            })
        .def("dp", [](fast_cd_arap_sim& sim) {
            fast_cd_arap_dynamic_precomp* p = (fast_cd_arap_dynamic_precomp*)sim.dp;
        return p;
            })
        .def("sol", [](fast_cd_arap_sim& sim) {
        fast_cd_arap_local_global_solver* p = (fast_cd_arap_local_global_solver*)sim.sol;
        return p;
        })
        ;

    py::class_<cd_sim_state>(m, "cd_sim_state")
        .def(py::init<>())
        .def(py::init<VectorXd&, VectorXd&, VectorXd&, VectorXd&>())
        .def(py::init<VectorXd&, VectorXd>())
        .def("init", static_cast<void (cd_sim_state::*)(const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&)>(&cd_sim_state::init))
        .def("init", static_cast<void (cd_sim_state::*)(const VectorXd&, const VectorXd&)>(&cd_sim_state::init))
        .def("update", static_cast<void (cd_sim_state::*)(const VectorXd&, const VectorXd&)>(&cd_sim_state::update))
        .def("update", static_cast<void (cd_sim_state::*)(const VectorXd&)>(&cd_sim_state::update))
        .def_readwrite("z_curr", &cd_sim_state::z_curr)
        .def_readwrite("z_prev", &cd_sim_state::z_prev)
        .def_readwrite("p_curr", &cd_sim_state::p_curr)
        .def_readwrite("p_prev", &cd_sim_state::p_prev);

    py::class_<fast_cd_sim_params>(m, "fast_cd_sim_params")
        .def(py::init<>())
        .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
            EigenDRef<MatrixXd>, const VectorXi&,
            const SparseMatrix<double>&, double, double, double,
            bool, string >(), " \n \
            Contains all the parameters required to build a \n \
            fast Complementary Dynamics simulator \n \
            Inputs: \n \
                X - n x 3 vertex geometry \n \
                T - T x 4 tet indices \n \
                B - 3n x m subspace matrix \n \
                l - T x 1 clustering labels for each tet \n \
                J - 3n x p rig jacobian \n \
                mu - (double)first lame parameter \n \
                lambda - (double)second lame parameter \n \
                do_inertia - (bool)whether or not sim should have inertia \n \
            (if no, then adds Tik.regularizer to laplacian \n \
                sim_constraint_type - (string) \"none\" or \"cd\" or \"cd_momentum_leak\" for now \n \
            ")
        .def_readwrite("X", &fast_cd_sim_params::X)
        .def_readwrite("T", &fast_cd_sim_params::T)
        .def_readwrite("B", &fast_cd_sim_params::B)
        .def_readwrite("labels", &fast_cd_sim_params::labels)
        .def_readwrite("do_inertia", &fast_cd_sim_params::do_inertia)
        .def_readwrite("Aeq", &fast_cd_sim_params::Aeq)
        .def_readwrite("h", &fast_cd_sim_params::h)
        .def_readwrite("invh2", &fast_cd_sim_params::invh2)
        .def_readwrite("mu", &fast_cd_sim_params::mu)
        .def_readwrite("lambda", &fast_cd_sim_params::lambda);

    py::class_<fast_cd_external_force>(m, "fast_cd_external_force", "A \
        class that holds common spatiotemporal control \
        forces we can use for test scenes in CD")
        .def(py::init<fast_cd_sim_params&, string, double>(), py::arg("params"),
            py::arg("external_force_type") = "none", py::arg("external_force_magnitude"), 
            " \
            Initializes external force used in simulation \n \
            sim_params - (fast_cd_sim_params)parameters of our simulation \n \
            external_force_type - (string)either \"none\", or \"momentum_leak\" \n \
            external_force_magnitude - (double) "
        )
        .def("get", &fast_cd_external_force::get, " \
        Returns the external force being supplied to the fast complementary dynamics system. \n \
        Inputs: \n \
            step - which timestep of the simulation are we in.This is useful for forces that have a time - varying component \n \
            p - 12 | B | x1 flattened rig parameters at next timestep \n \
            state - fast_cd_state struct that contains info on z_curr, z_prev,\
                        p_currand p_prev.Useful for inertial - like external forces \n \
            Output - \n \
            f - m x 1 external force at this timestep \n \
            ");

    py::class_<cd_arap_local_global_solver_params>(m, "cd_arap_local_global_solver_params", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<bool, int, double>());
       
    py::class_<fast_cd_arap_static_precomp>(m, "fast_cd_arap_static_precomp")
        .def(py::init<>());

    py::class_<fast_cd_arap_dynamic_precomp>(m, "fast_cd_arap_dynamic_precomp")
        .def(py::init<>());

    py::class_<fast_cd_viewer>(m, "fast_cd_viewer")
        .def(py::init<>())
        .def("set_mesh", &fast_cd_viewer::set_mesh)
        .def("set_vertices", &fast_cd_viewer::set_vertices)
        .def("invert_normals", &fast_cd_viewer::invert_normals)
        .def("set_camera_center", &fast_cd_viewer::set_camera_center)
        .def("set_camera_eye", &fast_cd_viewer::set_camera_eye)
        .def("set_camera_zoom", &fast_cd_viewer::set_camera_zoom)
        .def("clear", &fast_cd_viewer::clear)
        .def("compute_normals", &fast_cd_viewer::compute_normals)
        .def("add_mesh", [](fast_cd_viewer& v) {
        int id;
        v.add_mesh(id);
        return id; })
        .def("add_mesh", [](fast_cd_viewer& v, EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F) {
            int id;
            v.add_mesh(V, F, id);
            return id; })
        .def("set_pre_draw_callback", [&](fast_cd_viewer& v, std::function<void(void)>& func)
            {
                auto wrapperFunc = [=](igl::opengl::glfw::Viewer&) -> bool {
                    func();
                    return false;
                };
                v.igl_v->callback_pre_draw = wrapperFunc;
      
        })
         
        .def("set_key_callback", &fast_cd_viewer::set_key_pressed_callback)
        .def("set_color", static_cast<void (fast_cd_viewer::*)(const RowVector3d&, int)>(&fast_cd_viewer::set_color))
        .def("set_color", static_cast<void (fast_cd_viewer::*)(const MatrixXd&, int)>(&fast_cd_viewer::set_color))
        .def("set_show_faces", &fast_cd_viewer::set_show_faces)
        .def("set_show_lines", &fast_cd_viewer::set_show_lines)
        .def("launch", &fast_cd_viewer::launch)
        .def("init_guizmo", [&](fast_cd_viewer& v, bool visible, EigenDRef<Matrix4f> A0,  std::function<void(const Matrix4f &)> func, std::string operation)
        {
               
                v.guizmo->visible = visible;
                v.guizmo->T = A0;
                auto wrapperFunc = [=](const Matrix4f& A ) {
                    func(A);
                };
                v.guizmo->callback = wrapperFunc;

                if (operation == "scale")
                    v.guizmo->operation = ImGuizmo::SCALE;
                if (operation == "translate")
                    v.guizmo->operation = ImGuizmo::TRANSLATE;
                if (operation == "rotate")
                    v.guizmo->operation = ImGuizmo::ROTATE;
        })
        .def("change_guizmo_op", [&](fast_cd_viewer& v, std::string operation)
            {
                if (operation == "scale")
                    v.guizmo->operation = ImGuizmo::SCALE;
                if (operation == "translate")
                    v.guizmo->operation = ImGuizmo::TRANSLATE;
                if (operation == "rotate")
                    v.guizmo->operation = ImGuizmo::ROTATE;
            });
       // .def("set_points", &fast_cd_viewer::set_points)

        m.def("compute_clusters_weight_space", [](EigenDRef<MatrixXi> T, EigenDRef<MatrixXd> B, EigenDRef<VectorXd> L, int num_clusters, int num_feature_modes)
            {
                VectorXi labels;
                MatrixXd C;
                compute_clusters_weight_features(T, B, L, num_clusters, num_feature_modes, labels, C);
                return std::make_tuple(labels, C);
                    });
                m.def("lbs_jacobian", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXd> W) {
                    SparseMatrix<double> J;
                lbs_jacobian(V, W, J);
                return J;
            });

    /// INDEPENDANT FUNCTIONS 
        m.def("get_modes", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> T,
            SparseMatrix<double>& J, std::string mode_type, int num_modes) {
                MatrixXd B, Ws;
        VectorXd L;

        if (J.cols() != V.rows() * V.cols())
        {
            printf("Constraint matrix does not \
                have enough columns! Should be a #V*dim x #params matrix");
            return std::make_tuple(B, Ws, L);
        }
        else
        {
            get_modes(V, T, J, mode_type, num_modes, B, L, Ws);
            return std::make_tuple(B, Ws, L);
        }}
        , "hi"); /*
    , "Computes subspace used for fast_cd simulation \n \
Inputs: \n V - n x 3 mesh geometry \n T - T x 4 tet indices \n \
J - p x 3V subspace constraint matrix \n mode_type - (string) either \"skinning\" \
or \"displacement\" \n num_modes : (int) number of modes to compute in our subspace \n \
Outputs : \n B - 3n x num_modes full subspace columns\n W - n x num_modes \
skinning weights (empty if mode_type == \"displacement\" ) \n L - num_modes x 1 list of\
 eigenvalues sorted to match with the columns of B or W"
    //(string)either \"none\", \"cd\" or \"none\"

    m.def("skinning_modes", [](EigenDRef<MatrixXd> V, 
        SparseMatrix<double>& H, SparseMatrix<double>& M,
        SparseMatrix<double>& Aeq, int num_modes)
        {
            MatrixXd B_lbs, W;
            VectorXd L;
            skinning_modes(V, H, M, Aeq, num_modes, B_lbs, W, L);
            return std::make_tuple(B_lbs, W, L);
        });
    /*
    m.def("compute_clusters", [](EigenDRef<MatrixXi> T, EigenDRef<MatrixXd> B, EigenDRef<VectorXd> L, int num_clusters, int num_feature_modes)
        {
            VectorXi labels;
            MatrixXd C;
            compute_clusters_igl(T, B, L, num_clusters, num_feature_modes, labels, C);
            return std::make_tuple(labels, C);
        });
        */


    m.def("momentum_leaking_matrix", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> T) {
        SparseMatrix<double> D;
        momentum_leaking_matrix(V, T, fast_cd::MOMENTUM_LEAK_DIFFUSION, D);
        return D;
        });

    m.def("scale_and_center_geometry", [](EigenDRef<MatrixXd> V, const double h, EigenDRef<RowVector3d> c)
        {
            RowVector3d to;
            double so;
            MatrixXd V2 = scale_and_center_geometry(V, h, c, so, to);
            return std::make_tuple(V2, so, to);
        });
    

    m.def("massmatrix", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F) {
        SparseMatrix<double> M;

        igl::massmatrix(V, F,igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
        return M;
        });


    m.def("get_skeleton_mesh", [](double thickness, const VectorXd& p, const VectorXd& bl) {
        float t = (float) thickness;
        MatrixXd renderV, renderC;
        MatrixXi renderF;
        get_skeleton_mesh( thickness,  p,  bl,  renderV,  renderF,  renderC);
        return std::make_tuple(renderV, renderF, renderC);
        });



    m.def("get_skeleton_mesh", [](double thickness,const  VectorXd&  p, const VectorXd&  p_w, const VectorXd& bl) {
        float t = (float)thickness;
        MatrixXd renderV, renderC;
        MatrixXi renderF;
        get_skeleton_mesh(thickness, p, p_w, bl, renderV, renderF, renderC);
        return std::make_tuple(renderV, renderF, renderC);
        });


    //LIBIGL WRAPPERS
    m.def("readMSH", [](std::string filename) {
        MatrixXd V; MatrixXi F, T;
        VectorXi tritag, tettag;
        igl::readMSH(filename, V, F, T, tritag, tettag);
        return std::make_tuple(V, F, T);
        });
    
    m.def("readOBJ", [](std::string filename) {
        MatrixXd V; MatrixXi F;
        igl::readOBJ(filename, V, F); 
        return std::make_tuple(V, F);
        });

    m.def("read_rig_from_json", [](std::string filename) {
        MatrixXd V; MatrixXi F;
        MatrixXd W; MatrixXd P0; VectorXi pI; VectorXd l;
        std::string rig_type;
        read_rig_from_json(filename, W, P0, pI, l, V, F, rig_type);

        return std::make_tuple(W, P0, pI, l, V, F, rig_type);
        }, " \n \
        Reads rig from json file \n \
            Inputs : \n \
        rig_path - (string)where is my rig.json file ? \n \
            Outputs : \n \
            W - n x m matrix of rig weights \n \
            P0 - 4mx3 matrix of rest bone world transforms  \
            , where each 4x3 block is an affine handle \n \
            pI - mx1 parent bone Indicies, in the case of a skeleton rig \n \
            l - mx1 bone lengths, useful for visualization \n \
            V - nx 3 mesh geometry \n \
            F - Fx 3 | 4 mesh face / tet indices \n \
            rig_type - either \"surface\" or \"volume\", \n \
        ");

    m.def("surface_to_volume_weights", [](EigenDRef<MatrixXd> Ws,
        EigenDRef<MatrixXd> Vs, EigenDRef<MatrixXd>
        V, EigenDRef<MatrixXi> T) {
            MatrixXd W = surface_to_volume_weights(Ws, Vs, V, T);
    return W;
        }, " \n \
        Transfers weights defined on the surface Vs to the volume V \n \
        Fits the surface vertices of a mesh b(V) \n \
                to the surface points Vs \n \
        via a procrustes fit. \n \
             \n \
        Inputs: \n \
        surfaceW - bn x m set of surface weights defined over the mesh \n \
            surfaceV - bn x 3 surface vertices vertex positions \n \
            X - n x 3 full mesh vertices \n \
            T - T x 4 tet mesh indices \n \
            Outputs : \n \
        W - n x m set of surface weights obtained via a diffusion \n \
            from the surface to the interior \n \
        ");

    m.def("fit_rig_to_mesh_surface", [](
        EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> T,
        EigenDRef<MatrixXd> Vs, EigenDRef<MatrixXd> P0) {
            MatrixXd B;
        MatrixXd P1 = P0;
            fit_rig_to_mesh(V, T, Vs, P1, B);
            return std::make_tuple(P1, B);
        }, " \n \
    Given a surface rig with parameters P0,  \
        fit it to the mesh defined by V, T.  \
    The new rig bones might be changing scale, \
        rotationand centroid....  \
            Inputs : \n \
            V - mesh geometry \n \
            T - tet indices \n \
            Vs - intiial surface geometry associated with rig \n \
            P0 - initial rig parameters \n \
            lengths - intiial rig lengths \n \
        Outpus : \n \
            P0 - fitted rig parameters \n \
            B - the affine matrix that fits the rig to the mesh \n \
        ");

    m.def("fit_rig_to_mesh", [](
        EigenDRef<MatrixXd> V, 
        EigenDRef<MatrixXd> Vs, EigenDRef<MatrixXd> P0) {
            MatrixXd B;
    MatrixXd P1 = P0;
    fit_rig_to_mesh_vertices(V, Vs, P1, B);
    std::make_tuple(P1, B);
        }, " \n \
    Given a surface rig with parameters P0,  \
        fit it to the mesh defined by V, T.  \
    The new rig bones might be changing scale, \
        rotationand centroid....  \
            Inputs : \n \
            V - mesh geometry \n \
            Vs - intiial surface geometry associated with rig \n \
            P0 - initial rig parameters \n \
            lengths - intiial rig lengths \n \
        Outpus : \n \
            P0 - fitted rig parameters \n \
            B - the affine matrix that fits the rig to the mesh \n \
        ");

    m.def("read_anim_from_json", [](string anim_path) {
        MatrixXd animP;
        read_rig_anim_from_json(anim_path, animP);
        return animP;
        }, " \n \
        Reads a rig animation from a json file. \n \
            Input: \n \
        anim_path - path to anim.json file \n \
            Output : \n \
        animP - 12m x #frames matrix containing the world  \
            space animations bone transformations \
            for each frame \n \
        ");

    m.def("transform_rig_parameters_anim", [](EigenDRef<MatrixXd> animP,
        EigenDRef<MatrixXd> A) {
        MatrixXd P = animP;
        MatrixXd B = A;
        transform_rig_parameters_anim(P, B);
        return P;
        }, " \
        Given a world space set of animation frames for rig parameters P_w, \
            and a worlds space rest pose rig parameters p0, \
        compute the animation as relative space rig parameters, as needed for CD \
        ");

    m.def("world_to_rel_rig_anim", [](EigenDRef<MatrixXd> animPw,
        VectorXd& p0) {
            MatrixXd animP;
            world_to_rel_rig_anim(animPw, p0, animP);
            return animP;
        });

    m.def("writeDMATd", [](std::string filename, EigenDRef<MatrixXd> D) {
        return igl::writeDMAT(filename, D);
        });
    
    m.def("writeDMATi", [](std::string filename, EigenDRef<MatrixXi> D) {
        return igl::writeDMAT(filename, D);
        });

    m.def("boundary_facets", [](EigenDRef<MatrixXi> T) {
        MatrixXi F;
    VectorXi FiT, K;
    igl::boundary_facets(T, F, FiT, K);
    return std::make_tuple( F, FiT, K);
     });
        

}