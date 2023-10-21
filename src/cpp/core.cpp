#include "fast_cd_arap_sim.h"
#include "rig_parameters.h"
#include "get_skeleton_mesh.h"
#include "skinning_modes.h"
#include "compute_clusters_igl.h"

#include "lbs_jacobian.h"
#include "scale_and_center_geometry.h"
#include "read_fast_cd_sim_static_precompute.h"
#include "write_fast_cd_sim_static_precomputation.h"
#include "surface_to_volume_weights.h"
#include "fit_rig_to_mesh.h"
#include "momentum_leaking_matrix.h"
#include "read_rig_from_json.h"
#include "write_rig_to_json.h"
#include "fast_cd_subspace.h"
#include "read_rig_anim_from_json.h"
#include "rig_parameters.h"
#include "selection_matrix.h"
#include "prolongation.h"
//#include "fast_cd_scene.h"
#include "vector_gradient_operator.h"
#include "fast_cd_corot_sim_params.h"
#include "fast_cd_corot_sim.h"
#include "compute_bbw_weights.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;
using namespace std;


// Forward-declare bindings from other files
void bind_viewer(py::module& m);
void bind_igl(py::module& m);

void bind_fast_cd_arap_sim(py::module& m)
{
    py::class_<fast_cd_arap_local_global_solver>(m, "fast_cd_arap_local_global_solver")
        .def(py::init<>())
        .def(py::init<EigenDRef<MatrixXd>,
            EigenDRef<MatrixXd>, local_global_solver_params&>(), " \n \
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


    py::class_<cd_arap_sim>(m, "cd_arap_sim")
        .def(py::init<>())
        .def(py::init<cd_sim_params&, local_global_solver_params&>())
        .def("step", static_cast<VectorXd(cd_arap_sim::*)(
            const VectorXd&, const VectorXd&, const cd_sim_state&,
            const  VectorXd&, const  VectorXd&)>
            (&cd_arap_sim::step))
        .def("step", static_cast<VectorXd(cd_arap_sim::*)(
            const VectorXd&, const cd_sim_state&,
            const  VectorXd&, const  VectorXd&)>
            (&cd_arap_sim::step))
        .def("set_equality_constraint", &cd_arap_sim::set_equality_constraint)
        .def("params", &cd_arap_sim::parameters)
        ;

    py::class_<cd_sim_params>(m, "cd_sim_params")
        .def(py::init<>())
        .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
            const SparseMatrix<double>&, double, double, double,
            bool, string>())
        .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
            double, double, double,
            bool, string>())
        .def_readwrite("X", &cd_sim_params::X)
        .def_readwrite("T", &cd_sim_params::T)
        .def_readwrite("do_inertia", &cd_sim_params::do_inertia)
        .def_readwrite("Aeq", &cd_sim_params::Aeq)
        .def_readwrite("h", &cd_sim_params::h)
        .def_readwrite("invh2", &cd_sim_params::invh2)
        .def_readwrite("mu", &cd_sim_params::mu)
        .def_readwrite("lambda", &cd_sim_params::lambda);
    ;



    py::class_<fast_cd_arap_sim>(m, "fast_cd_arap_sim")
        .def(py::init<std::string&, fast_cd_arap_sim_params&,
            local_global_solver_params&, bool, bool>())
        .def(py::init<fast_cd_arap_sim_params&, local_global_solver_params&>(), "\
            cache_dir - (string)  \n \
            sim_params - fast_cd_arap_sim_params  \n \
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
            (&fast_cd_arap_sim::step), " \n \
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
        fast_cd_arap_sim_params* p = (fast_cd_arap_sim_params*)sim.params;
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
        .def("set_equality_constraint", &fast_cd_arap_sim::set_equality_constraint)
        ;


        py::class_<fast_cd_arap_sim_params>(m, "fast_cd_arap_sim_params")
            .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
                EigenDRef<MatrixXd>, const VectorXi&,
                const SparseMatrix<double>&, const SparseMatrix<double>&, 
                double, double, double >(), " \n \
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
            .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
                EigenDRef<MatrixXd>, const VectorXi&,
                const SparseMatrix<double>&, const SparseMatrix<double>&,
                const VectorXd&, double, double >())
            .def_readwrite("X", &fast_cd_arap_sim_params::X)
            .def_readwrite("T", &fast_cd_arap_sim_params::T)
            .def_readwrite("B", &fast_cd_arap_sim_params::B)
            .def_readwrite("labels", &fast_cd_arap_sim_params::labels)
            .def_readwrite("rho", &fast_cd_arap_sim_params::rho)
            .def_readwrite("Aeq", &fast_cd_arap_sim_params::Aeq)
            .def_readwrite("h", &fast_cd_arap_sim_params::h)
            .def_readwrite("invh2", &fast_cd_arap_sim_params::invh2)
            .def_readwrite("mu", &fast_cd_arap_sim_params::mu);

    

        py::class_<local_global_solver_params>(m, "local_global_solver_params", py::dynamic_attr())
            .def(py::init<>())
            .def(py::init<bool, int, double>())
            .def_readwrite("max_iters", &local_global_solver_params::max_iters)
            .def_readwrite("threshold", &local_global_solver_params::threshold)
            .def_readwrite("to_convergence", &local_global_solver_params::to_convergence);
        

        py::class_<fast_cd_arap_static_precomp>(m, "fast_cd_arap_static_precomp")
            .def(py::init<>());

        py::class_<fast_cd_arap_dynamic_precomp>(m, "fast_cd_arap_dynamic_precomp")
            .def(py::init<>());

}

void bind_fast_cd_corot_sim(py::module& m)
{
    py::class_<fast_cd_corot_sim>(m, "fast_cd_corot_sim")
        //.def(py::init<std::string&, fast_cd_corot_sim_params&,
        //    local_global_solver_params&, bool, bool>())
        .def(py::init<fast_cd_corot_sim_params&, local_global_solver_params&>())
        .def("step", static_cast<VectorXd(fast_cd_corot_sim::*)(
            const VectorXd&, const VectorXd&, const cd_sim_state&,
            const  VectorXd&, const  VectorXd&)>
            (&fast_cd_corot_sim::step), " \ \
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
        .def("params", [](fast_cd_corot_sim& sim) {
        fast_cd_corot_sim_params* p = (fast_cd_corot_sim_params*)sim.params;
    return p;
            })
        .def("sp", [](fast_cd_corot_sim& sim) {
                fast_cd_corot_static_precomp* p = (fast_cd_corot_static_precomp*)sim.sp;
            return p;
            })
        .def("dp", [](fast_cd_corot_sim& sim) {
        fast_cd_corot_dynamic_precomp* p = (fast_cd_corot_dynamic_precomp*)sim.dp;
        return p;
        })
        .def("sol", [](fast_cd_corot_sim& sim) {
        fast_cd_corot_local_global_solver* p = (fast_cd_corot_local_global_solver*)sim.sol;
        return p;
        })
        .def("set_equality_constraint", &fast_cd_corot_sim::set_equality_constraint)
        ;


        py::class_<fast_cd_corot_sim_params>(m, "fast_cd_corot_sim_params")
            .def(py::init<>())
            .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
                EigenDRef<MatrixXd>, const VectorXi&,
                const SparseMatrix<double>&, double, double, double,
                bool >(), " \n \
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
            .def_readwrite("X", &fast_cd_corot_sim_params::X)
            .def_readwrite("T", &fast_cd_corot_sim_params::T)
            .def_readwrite("B", &fast_cd_corot_sim_params::B)
            .def_readwrite("labels", &fast_cd_corot_sim_params::labels)
            .def_readwrite("do_inertia", &fast_cd_corot_sim_params::do_inertia)
            .def_readwrite("Aeq", &fast_cd_corot_sim_params::Aeq)
            .def_readwrite("h", &fast_cd_corot_sim_params::h)
            .def_readwrite("invh2", &fast_cd_corot_sim_params::invh2)
            .def_readwrite("mu", &fast_cd_corot_sim_params::mu)
            .def_readwrite("lambda", &fast_cd_corot_sim_params::lambda);

}

PYBIND11_MODULE(fast_cd_pyb, m) {
 /*   py::class_<fast_cd_scene_obj>(m, "fast_cd_scene_obj", py::dynamic_attr())
        .def(py::init<std::string, EigenDRef<MatrixXd>, EigenDRef<MatrixXi>, std::string, std::string, fast_cd_subspace&, fast_cd_arap_sim&>())
        .def("transform_animation", &fast_cd_scene_obj::transform_animation)
        .def("step", &fast_cd_scene_obj::step)
        .def_readwrite("do_cd", &fast_cd_scene_obj::do_cd, py::return_value_policy::reference_internal)
        ;

    py::class_<fast_cd_scene>(m, "fast_cd_scene")
        .def(py::init<std::string, std::string, int, int>(), py::arg("vert_shader_path"),
            py::arg("frag_shader_path"), py::arg("max_primary_bones") = 16, py::arg("max_secondary_bones") = 16)
        .def("add_scene_object", static_cast<int(fast_cd_scene::*)(
            fast_cd_scene_obj&)>
            (&fast_cd_scene::add_scene_object))
        .def("add_scene_object", static_cast<int(fast_cd_scene::*)(
            fast_cd_scene_obj&, fast_cd_viewer_parameters&)>
            (&fast_cd_scene::add_scene_object))
        .def("show", &fast_cd_scene::show)
        .def_readwrite("num_v", &fast_cd_scene::num_v)
        .def_readwrite("num_t", &fast_cd_scene::num_t)
        .def_readwrite("num_obj", &fast_cd_scene::num_obj)
        .def("set_do_cd", &fast_cd_scene::set_do_cd)
        .def("set_background_color", &fast_cd_scene::set_background_color)
        .def("set_record", &fast_cd_scene::set_record)
        ;*/

    py::class_<sim_state>(m, "sim_state",
        "Simulation state \ \n")
        .def(py::init<>())
        .def(py::init<const VectorXd&, const VectorXd&>())
        .def("update",  & sim_state::update)
        ;




    py::class_<fast_cd_subspace>(m, "fast_cd_subspace", "Helper class that \
        builds/reads/writes the subspaces necessary to run and test Fast CD\n",
        py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<EigenDRef<MatrixXd>,EigenDRef<MatrixXd>, VectorXi,string>() )
        .def(py::init<string, string, string, int, int>() )
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
        .def("setlabels", [](fast_cd_subspace& sub, VectorXi& l) {sub.l = l; })
        .def_readwrite("B", &fast_cd_subspace::B, py::return_value_policy::reference_internal)
        .def_readwrite("W", &fast_cd_subspace::W, py::return_value_policy::reference_internal)
        .def_readwrite("labels", &fast_cd_subspace::l, py::return_value_policy::reference_internal)
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


        m.def("complementarity_constraint_matrix",[](EigenDRef<MatrixXd> V, EigenDRef<MatrixXd> W,
            EigenDRef<MatrixXi> T)
        {
            SparseMatrix<double> J, D, M, Aeq;
            lbs_jacobian(V, W, J);
            igl::massmatrix(V, T, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
            Aeq = (J.transpose() * igl::repdiag(M, 3));
            return Aeq;
        });

        m.def("complementarity_constraint_matrix_with_diffusion_leaking", [](EigenDRef<MatrixXd> V, 
            EigenDRef<MatrixXd> W, EigenDRef<MatrixXi> T,  double dt, bool flip) {
            SparseMatrix<double> J, D, M, Aeq;
            lbs_jacobian(V, W, J);
            
            if (flip)
                momentum_leaking_matrix(V, T, fast_cd::MOMENTUM_LEAK_DIFFUSION_FLIP, D, dt);
            else
                momentum_leaking_matrix(V, T, fast_cd::MOMENTUM_LEAK_DIFFUSION, D, dt);

            igl::massmatrix(V, T, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
            Aeq = (J.transpose() * igl::repdiag(M, 3) * igl::repdiag(D, 3));
            return Aeq;
            });

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

    m.def("write_rig_to_json", [](std::string filename,
        EigenDRef<MatrixXd> W, EigenDRef<MatrixXd> P0, VectorXi& pI, VectorXd& l,
        EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F,
    std::string rig_type) {
    return write_rig_to_json(filename, W, P0, pI, l, V, F, rig_type);
        });

    m.def("surface_to_volume_weights", [](EigenDRef<MatrixXd> Ws,
        EigenDRef<MatrixXd> Vs, EigenDRef<MatrixXd>
        V, EigenDRef<MatrixXi> T) {
          //  printf("Made it before surface_to_volume_weights call!\n");
            MatrixXd W = surface_to_volume_weights(Ws, Vs, V, T);
          //  printf("Made it past surface_to_volume_weights call!\n");
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


        
    m.def("selection_matrix", [](const VectorXi& bI, int n, int dim)
    {
        SparseMatrix<double> S;
        selection_matrix(bI, n, dim, S);
        return S;
    }, py::arg("bI"), py::arg("n"), py::arg("dim") = 1);

    m.def("prolongation", [](EigenDRef<MatrixXd>Xf, EigenDRef<MatrixXd> Xc, EigenDRef<MatrixXi> T)
    {
        SparseMatrix<double> P;
        prolongation(Xf, Xc, T, P);
        return P;
    });

    m.def("interweaving_matrix", [](int rows, int cols) {
        SparseMatrix<double>S;
        interweaving_matrix(rows, cols, S);
        return S;
        });

    m.def("vector_gradient_operator", [](EigenDRef<MatrixXd> X, EigenDRef<MatrixXi> T) {
        SparseMatrix<double> K;
        vector_gradient_operator(X, T, K);
        return K;
        });

    m.def("compute_bbw_weights", [](EigenDRef<VectorXd> p, EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> T)
        {
            MatrixXd W = compute_bbw_weights(p, V, T);
            return W;
        });


    bind_fast_cd_arap_sim(m);
    bind_fast_cd_corot_sim(m);
    bind_igl(m);
    bind_viewer(m);
}