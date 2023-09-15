//#include "fast_cd_arap_sim.h"
//#include "fast_cd_subspace.h"
//#include "fast_cd_external_force.h"
//#include "fast_ik_subspace.h"
//#include "fast_ik_sim.h"
//
//#include <pybind11/pybind11.h>
//#include <pybind11/eigen.h>
//#include <pybind11/functional.h>
//#include <string>
//namespace py = pybind11;
//using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
//template <typename MatrixType>
//using EigenDRef = Ref<MatrixType, 0, EigenDStride>;
//using namespace std;
//
//
//void bind_sim(py::module& m) 
// {
//    py::class_<fast_ik_subspace>(m, "fast_ik_subspace", "Helper Class that builds, reads and writes \
//         the FAST IK subspace\n")
//        .def(py::init<>())
//        .def(py::init<int, bool, string>(), py::arg("num_clusters"),
//            py::arg("debug") = false, py::arg("output_dir") = "")
//        .def("init", &fast_ik_subspace::init)
//        .def("init_with_cache", &fast_ik_subspace::init_with_cache)
//        .def("write_to_cache", &fast_ik_subspace::write_to_cache)
//        .def("read_from_cache", &fast_ik_subspace::read_from_cache)
//        .def("read_from_cache_recompute", &fast_ik_subspace::read_from_cache_recompute)
//        .def_readwrite("labels", &fast_ik_subspace::l)
//        ;
//
//
//    /*MatrixXd& V, MatrixXi& T, MatrixXd& W,
//        VectorXi& l, VectorXi& bI,
//        cd_arap_local_global_solver_params& solver_params*/
//    py::class_<fast_ik_sim>(m, "fast_ik_sim", "FAST IK Simulation")
//        .def(py::init<>())
//        .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
//            EigenDRef<MatrixXd>, const VectorXi&,
//            SparseMatrix<double>&,
//            const local_global_solver_params&>())
//        .def("step", static_cast<VectorXd(fast_ik_sim::*)(
//            const VectorXd&,
//            const sim_state&, const VectorXd&, const VectorXd&)>
//            (&fast_ik_sim::step), " \n \
//            Advances the pre - configured simulation one step \n \
//            Inputs : \n \
//                z:  m x 1 current guess for z(maybe shouldn't expose this) \n \
//                sim_state : state  \n \
//                f_ext : m x 1 previos d.o.f.s  \n \
//                bc :  rhs of equality constraint if some are configured in system \n \
//                (should match in rows with sim.params.Aeq)  \n \
//            Outputs :  \n \
//                z_next: m x 1 next timestep degrees of freedom  \n \
//            ")
//        .def("set_equality_constraint", &fast_ik_sim::set_equality_constraint)
//        .def("params", [](fast_ik_sim& sim) {
//        fast_cd_arap_sim_params* p = (fast_cd_arap_sim_params*)sim.params;
//    return p;
//            })
//        ;
//
//            py::class_<fast_cd_subspace>(m, "fast_cd_subspace", "Helper class that \
//        builds/reads/writes the subspaces necessary to run and test Fast CD\n",
//                py::dynamic_attr())
//                .def(py::init<>())
//                .def(py::init<int, string, string, int, int, bool, bool, string>(), "\
//            	Initializes and configures subspace... does NOT build it yet.\n \
//                Inputs: \n \
//                num_modes - (int) number of modes in subspace\n \
//                subspace_constraint_type - (string) either \"none\", or \"cd\" or \
//                                            \"cd_momentum_leak\"\n \
//                mode_type - (string) either \"skinning\" or \"displacement\"\n \
//                num_clusters - (int) number of clusters\n \
//                num_clustering_features - (int) number of clustering features used\n \
//                split_components - (bool) whether to split the components of the clusters \n \
//                \n  \
//                Optional:\n\
//                debug - (bool) whether to save and store debug info\n \
//                output_dir - (string) output directory where we will write debug info\n\
//            ", py::arg("num_modes"), py::arg("subspace_constraint_type"),
//                    py::arg("mode_type"), py::arg("num_clusters"),
//                    py::arg("num_clustering_features"),
//                    py::arg("split_components"),
//                    py::arg("debug") = false, py::arg("output_dir") = "")
//                .def("init", &fast_cd_subspace::init, "\
//            Computes modes + clusters from scratch \n \
//            Inputs : \n \
//                V -> | n | x3 geometry \n \
//                T -> | T | x4 tet indices \n \
//                J -> | c | x3n null space / linear orthogonality constraint")
//                .def("init_with_cache", &fast_cd_subspace::init_with_cache, "\
//            Computes modes + clusters from scratch, \
//            with control over when to read / write from cache \n \
//            Inputs:  \n \
//                V -> | n | x3 geometry  \n \
//                T -> | T | x4 tet indices \n \
//                J -> 3n x | c | rig jacobian (not weighed by mass matrix or nothing) \n \
//                read_cache->whether or not to \
//                            attempt to read modes and clusters from cache. \n \
//                write_cache->whether or not to write modes and clusters to cache. \n \
//                modes_cache_dir->directory where mode cache is \n \
//            (Optional) \n \
//                clusters_cache_dir->directory where clusters cache is \n \
//                recompute_modes_if_not_found->whether or \
//                                            not to recompute modes from scratch \
//                                            if not found in cache(default true) \n \
//                recompute_clusters_if_not_found->whether or not to  \
//                                                recompute clusters from scratich if \
//                                                not found in cache(default true) \n \
//            ")
//                .def("write_to_cache", &fast_cd_subspace::write_to_cache, " \
//                Writes modes and clusters to cache directories \n \
//                modes_dir -> (string)where to save modes directory \
//                             (both B.DMAT or W.DMAT and L.DMAT, modes + frequencies) \n \
//                clusters_dir -> (string)where to save clusters directory(labels.DMAT)\
//                ")
//
//                .def("read_from_cache", &fast_cd_subspace::read_from_cache, "  \
//                Reads modes and clusters from cache directories \n \
//                Inputs: \n \
//                    modes_dir - (string)directory where B.DMAT / W.DMAT \n \
//                                    and L.DMAT is stored \
//                        clusters_dir - (string) directory where \
//                clusters labels.DMAT is stored \n \
//                ")
//                .def("read_clusters_from_cache", &fast_cd_subspace::read_clusters_from_cache,
//                    "  \
//                Read clusters from cache directories \n  \
//                Inputs: n \
//                    clusters_dir - (string)directory where cluster labels.DMAT is stored \n \
//                        ")
//                .def("read_modes_from_cache", &fast_cd_subspace::read_modes_from_cache,
//                    "  \
//                    Read clusters from cache directories \n \
//                    Inputs: \n \
//                        modes_dur - (string)directory where \
//                                        cluster B.DMAT / W.DMAT and L.DMAT is stored\
//                     ")
//                .def("getB", [](fast_cd_subspace& sub) {return sub.B; })
//                .def("getW", [](fast_cd_subspace& sub) {return sub.W; })
//                .def("getL", [](fast_cd_subspace& sub) {return sub.L; })
//                .def("getLabels", [](fast_cd_subspace& sub) {return sub.l; })
//                .def("setB", [](fast_cd_subspace& sub, EigenDRef<MatrixXd> B) {sub.B = B; })
//                .def("setW", [](fast_cd_subspace& sub, EigenDRef<MatrixXd> W) {sub.W = W; })
//                .def("setL", [](fast_cd_subspace& sub, VectorXd& L) {sub.L = L; })
//                .def("setlabels", [](fast_cd_subspace& sub, VectorXi& l) {sub.l = l; })
//                .def_readwrite("B", &fast_cd_subspace::B, py::return_value_policy::reference_internal)
//                .def_readwrite("W", &fast_cd_subspace::W, py::return_value_policy::reference_internal)
//                .def_readwrite("L", &fast_cd_subspace::L, py::return_value_policy::reference_internal)
//                .def_readwrite("labels", &fast_cd_subspace::l, py::return_value_policy::reference_internal)
//                ;
//
//
//
//            py::class_<fast_cd_arap_local_global_solver>(m, "fast_cd_arap_local_global_solver")
//                .def(py::init<>())
//                .def(py::init<EigenDRef<MatrixXd>,
//                    EigenDRef<MatrixXd>, local_global_solver_params&>(), " \n \
//            Constructs arap local global solver object used to solve \n \
//            dynamics quickly for fast Complementary DYnamoics \n \
//            Inputs : \n \
//                 A - m x m system matrix \n \
//                Aeq - c x m constraint rows that enforece Aeq z = b \n \
//                as linear equality constraints \n \
//                p - cd_arap_local_global_solver_params \n \
//            ")
//                .def(py::init<EigenDRef<MatrixXd>,
//                    EigenDRef<MatrixXd>, bool, int, double>(), " \n \
//            Constructs arap local global solver object used to solve \n \
//            dynamics quickly for fast Complementary DYnamoics \n \
//            Inputs : \n \
//                 A - m x m system matrix \n \
//                Aeq - c x m constraint rows that enforece Aeq z = b \n \
//                as linear equality constraints \n \
//               run_solver_to_convergence - (bool)  \n \
//                max_iters - (int) \n \
//                convergence_threshold - (double) \n \
//                            where to stop if || res || 2 drops below this \n \
//            ")
//                .def_readwrite("prev_solve_iters", &fast_cd_arap_local_global_solver::prev_solve_iters)
//                .def_readwrite("prev_res", &fast_cd_arap_local_global_solver::prev_res);
//
//
//            py::class_<cd_arap_sim>(m, "cd_arap_sim")
//                .def(py::init<>())
//                .def(py::init<cd_sim_params&, local_global_solver_params&>())
//                .def("step", static_cast<VectorXd(cd_arap_sim::*)(
//                    const VectorXd&, const VectorXd&, const cd_sim_state&,
//                    const  VectorXd&, const  VectorXd&)>
//                    (&cd_arap_sim::step))
//                .def("step", static_cast<VectorXd(cd_arap_sim::*)(
//                    const VectorXd&, const cd_sim_state&,
//                    const  VectorXd&, const  VectorXd&)>
//                    (&cd_arap_sim::step))
//                .def("set_equality_constraint", &cd_arap_sim::set_equality_constraint)
//                .def("params", &cd_arap_sim::parameters)
//                ;
//
//            py::class_<cd_sim_params>(m, "cd_sim_params")
//                .def(py::init<>())
//                .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
//                    const SparseMatrix<double>&, double, double, double,
//                    bool, string>())
//                .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
//                    double, double, double,
//                    bool, string>())
//                .def_readwrite("X", &cd_sim_params::X)
//                .def_readwrite("T", &cd_sim_params::T)
//                .def_readwrite("do_inertia", &cd_sim_params::do_inertia)
//                .def_readwrite("Aeq", &cd_sim_params::Aeq)
//                .def_readwrite("h", &cd_sim_params::h)
//                .def_readwrite("invh2", &cd_sim_params::invh2)
//                .def_readwrite("mu", &cd_sim_params::mu)
//                .def_readwrite("lambda", &cd_sim_params::lambda);
//            ;
//
//
//
//        py::class_<fast_cd_arap_sim>(m, "fast_cd_arap_sim")
//        .def(py::init<std::string&, fast_cd_arap_sim_params&,
//            local_global_solver_params&, bool, bool>())
//        .def(py::init<fast_cd_arap_sim_params&, local_global_solver_params&>(), "\
//        cache_dir - (string)  \n \
//        sim_params - fast_cd_arap_sim_params  \n \
//        solver_params - solver_params \n  \
//        read_cache - (bool) \n \
//        write_cache - (bool)\
//        ")
//        .def("step", static_cast<VectorXd(fast_cd_arap_sim::*)(
//            const VectorXd&, const VectorXd&, const cd_sim_state&,
//            const  VectorXd&, const  VectorXd&)>
//            (&fast_cd_arap_sim::step), " \ \
//        Advances the pre-configured simulation one step  \n \
//        Inputs : \n \
//        z:  m x 1 current guess for z(maybe shouldn't expose this) \n \
//        p : p x 1 flattened rig parameters following writeup column order flattening converntion \n \
//        state : sim_cd_state that contains info like z_curr, z_prev, p_currand p_prev \n \
//        f_ext : used to specify excternal forces like gravity. \n \
//        bc : rhs of equality constraint if some are configured in system \n \
//        (should match in rows with sim.params.Aeq) \n \
//        Outputs : \n \
//        z_next: m x 1 next timestep degrees of freedom \n \
//        ")
//        .def("step", static_cast<VectorXd(fast_cd_arap_sim::*)(
//            const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&,
//            const VectorXd&, const VectorXd&,
//            const  VectorXd&, const  VectorXd&)>
//            (&fast_cd_arap_sim::step), " \n \
//        Advances the pre - configured simulation one step \n \
//        Inputs : \n \
//        z:  m x 1 current guess for z(maybe shouldn't expose this) \n \
//        p : p x 1 flattened rig parameters following writeup column order flattening converntion  \n \
//        z_curr : m x 1 current d.o.f.s  \n \
//        z_prev : m x 1 previos d.o.f.s  \n \
//        p_curr : p x 1 current rig parameters  \n \
//        p_prev : p x 1 previous rig parameters  \n \
//        f_ext : used to specify excternal forces like gravity.  \n \
//        bc : rhs of equality constraint if some are configured in system  \n \
//        (should match in rows with sim.params.Aeq)  \n \
//        Outputs :  \n \
//        z_next: m x 1 next timestep degrees of freedom  \n \
//        ")
//        .def("params", [](fast_cd_arap_sim& sim) {
//        fast_cd_arap_sim_params* p = (fast_cd_arap_sim_params*)sim.params;
//        return p;
//            })
//        .def("sp", [](fast_cd_arap_sim& sim) {
//        fast_cd_arap_static_precomp* p = (fast_cd_arap_static_precomp*)sim.sp;
//        return p;
//        })
//        .def("dp", [](fast_cd_arap_sim& sim) {
//        fast_cd_arap_dynamic_precomp* p = (fast_cd_arap_dynamic_precomp*)sim.dp;
//        return p;
//        })
//        .def("sol", [](fast_cd_arap_sim& sim) {
//        fast_cd_arap_local_global_solver* p = (fast_cd_arap_local_global_solver*)sim.sol;
//        return p;
//        })
//        .def("set_equality_constraint", &fast_cd_arap_sim::set_equality_constraint)
//        ;
//
//py::class_<cd_sim_state>(m, "cd_sim_state")
//    .def(py::init<>())
//    .def(py::init<VectorXd&, VectorXd&, VectorXd&, VectorXd&>())
//    .def(py::init<VectorXd&, VectorXd>())
//    .def("init", static_cast<void (cd_sim_state::*)(const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&)>(&cd_sim_state::init))
//    .def("init", static_cast<void (cd_sim_state::*)(const VectorXd&, const VectorXd&)>(&cd_sim_state::init))
//    .def("update", static_cast<void (cd_sim_state::*)(const VectorXd&, const VectorXd&)>(&cd_sim_state::update))
//    .def("update", static_cast<void (cd_sim_state::*)(const VectorXd&)>(&cd_sim_state::update))
//    .def_readwrite("z_curr", &cd_sim_state::z_curr)
//    .def_readwrite("z_prev", &cd_sim_state::z_prev)
//    .def_readwrite("p_curr", &cd_sim_state::p_curr)
//    .def_readwrite("p_prev", &cd_sim_state::p_prev);
//
//    py::class_<fast_cd_arap_sim_params>(m, "fast_cd_arap_sim_params")
//        .def(py::init<>())
//        .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
//            EigenDRef<MatrixXd>, const VectorXi&,
//            const SparseMatrix<double>&, double, double, double,
//            bool, string >(), " \n \
//            Contains all the parameters required to build a \n \
//            fast Complementary Dynamics simulator \n \
//            Inputs: \n \
//                X - n x 3 vertex geometry \n \
//                T - T x 4 tet indices \n \
//                B - 3n x m subspace matrix \n \
//                l - T x 1 clustering labels for each tet \n \
//                J - 3n x p rig jacobian \n \
//                mu - (double)first lame parameter \n \
//                lambda - (double)second lame parameter \n \
//                do_inertia - (bool)whether or not sim should have inertia \n \
//            (if no, then adds Tik.regularizer to laplacian \n \
//                sim_constraint_type - (string) \"none\" or \"cd\" or \"cd_momentum_leak\" for now \n \
//            ")
//            .def_readwrite("X", &fast_cd_arap_sim_params::X)
//            .def_readwrite("T", &fast_cd_arap_sim_params::T)
//            .def_readwrite("B", &fast_cd_arap_sim_params::B)
//            .def_readwrite("labels", &fast_cd_arap_sim_params::labels)
//            .def_readwrite("do_inertia", &fast_cd_arap_sim_params::do_inertia)
//            .def_readwrite("Aeq", &fast_cd_arap_sim_params::Aeq)
//            .def_readwrite("h", &fast_cd_arap_sim_params::h)
//            .def_readwrite("invh2", &fast_cd_arap_sim_params::invh2)
//            .def_readwrite("mu", &fast_cd_arap_sim_params::mu)
//            .def_readwrite("lambda", &fast_cd_arap_sim_params::lambda);
//
//    py::class_<fast_cd_external_force>(m, "fast_cd_external_force", "A \
//    class that holds common spatiotemporal control \
//    forces we can use for test scenes in CD")
//        .def(py::init<fast_cd_arap_sim_params&, string, double>(), py::arg("params"),
//        py::arg("external_force_type") = "none", py::arg("external_force_magnitude"),
//            " \
//            Initializes external force used in simulation \n \
//            sim_params - (fast_cd_arap_sim_params)parameters of our simulation \n \
//            external_force_type - (string)either \"none\", or \"momentum_leak\" \n \
//            external_force_magnitude - (double) "
//        )
//        .def("get", &fast_cd_external_force::get, " \
//        Returns the external force being supplied to the fast complementary dynamics system. \n \
//        Inputs: \n \
//        step - which timestep of the simulation are we in.This is useful for forces that have a time - varying component \n \
//        p - 12 | B | x1 flattened rig parameters at next timestep \n \
//        state - fast_cd_state struct that contains info on z_curr, z_prev,\
//        p_currand p_prev.Useful for inertial - like external forces \n \
//        Output - \n \
//        f - m x 1 external force at this timestep \n \
//        ");
//
//        py::class_<local_global_solver_params>(m, "local_global_solver_params", py::dynamic_attr())
//            .def(py::init<>())
//            .def(py::init<bool, int, double>());
//
//        py::class_<fast_cd_arap_static_precomp>(m, "fast_cd_arap_static_precomp")
//            .def(py::init<>());
//
//        py::class_<fast_cd_arap_dynamic_precomp>(m, "fast_cd_arap_dynamic_precomp")
//            .def(py::init<>());
//
//
//             
//}