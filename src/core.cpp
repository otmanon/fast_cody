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
#include "fit_rig_to_mesh.h"
#include "momentum_leaking_matrix.h"

#include <igl/readMSH.h>
#include <igl/massmatrix.h>
#include <igl/readOBJ.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

PYBIND11_MODULE(fast_cd_pyb, m) {


    py::class_<fast_cd_arap_sim>(m, "fast_cd_arap_sim")
        .def(py::init<>())
        .def_readwrite("params", &fast_cd_arap_sim::params)
        .def_readwrite("sp", &fast_cd_arap_sim::sp)
        .def_readwrite("dp", &fast_cd_arap_sim::dp)
        .def(py::init<fast_cd_sim_params &, cd_arap_local_global_solver_params &>())
        .def(py::init<std::string &, cd_arap_local_global_solver_params &>())
        .def("save", &fast_cd_arap_sim::save)
        .def("step", static_cast<VectorXd(fast_cd_arap_sim::*)(
            const VectorXd &, const VectorXd &, const cd_sim_state& ,
            const  VectorXd & , const  VectorXd & )>
            (&fast_cd_arap_sim::step))
        .def("step_test", static_cast<VectorXd(fast_cd_arap_sim::*)(
            const VectorXd&, const VectorXd&, const cd_sim_state&,
            const  VectorXd&, const  VectorXd&)>
            (&fast_cd_arap_sim::step_test))
        .def("step", static_cast<VectorXd(fast_cd_arap_sim::*)(
            const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&,
            const VectorXd&, const VectorXd&,
            const  VectorXd&, const  VectorXd&)>
            (&fast_cd_arap_sim::step));

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
            EigenDRef<MatrixXd>, EigenDRef<VectorXi>,
            SparseMatrix<double>&, double, double, double,
            bool >())
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

    py::class_<cd_arap_local_global_solver_params>(m, "cd_arap_local_global_solver_params")
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


    /// INDEPENDANT FUNCTIONS 
    m.def("get_modes", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> T, EigenDRef<MatrixXd> W, SparseMatrix<double>& J, std::string mode_type, int num_modes ) {
        MatrixXd B, Ws;
        VectorXd L;
        get_modes(V, T, W, J, mode_type, num_modes, B, L, Ws);
        return std::make_tuple(B, Ws, L);
        });

    m.def("skinning_modes", [](EigenDRef<MatrixXd> V, SparseMatrix<double>& H, SparseMatrix<double>& M, SparseMatrix<double>& Aeq, int num_modes)
        {
            MatrixXd B_lbs, W;
            VectorXd L;
            skinning_modes(V, H, M, Aeq, num_modes, B_lbs, W, L);
            return std::make_tuple(B_lbs, W, L);
        });

    m.def("compute_clusters", [](EigenDRef<MatrixXi> T, EigenDRef<MatrixXd> B, EigenDRef<VectorXd> L, int num_clusters, int num_feature_modes)
        {
            VectorXi labels;
            MatrixXd C;
            compute_clusters_igl(T, B, L, num_clusters, num_feature_modes, labels, C);
            return std::make_tuple(labels, C);
        });

    m.def("compute_clusters_weight_space", [](EigenDRef<MatrixXi> T, EigenDRef<MatrixXd> B, EigenDRef<VectorXd> L, int num_clusters, int num_feature_modes)
        {
            VectorXi labels;
            MatrixXd C;
            compute_clusters_weight_space(T, B, L, num_clusters, num_feature_modes, labels, C);
            return std::make_tuple(labels, C);
        });
    m.def("lbs_jacobian", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXd> W) {
        SparseMatrix<double> J;
        lbs_jacobian(V, W, J);
        return J;
        });

    m.def("momentum_leaking_matrix", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> T) {
        SparseMatrix<double> D;
        momentum_leaking_matrix(V, T, fast_cd::MOMENTUM_LEAK_DIFFUSION, D);
        return D; });

    m.def("scale_and_center_geometry", [](EigenDRef<MatrixXd> V, const double h, EigenDRef<RowVector3d> c)
        {
            RowVector3d to;
            double so;
            MatrixXd V2 = scale_and_center_geometry(V, h, c, so, to);
            return std::make_tuple(V2, so, to);
        });
    
    m.def("read_fast_cd_sim_static_precomputation", [](std::string cache_dir) {
        fast_cd_arap_static_precomp sp;
        MatrixXd B; VectorXd L; VectorXi l;
        read_fast_cd_sim_static_precomputation(cache_dir, B, L, l, sp.BCB, sp.BMB, sp.BAB,
            sp.AeqB, sp.GmKB, sp.GmKJ, sp.GmKx, sp.G1VKB, sp.BMJ, sp.BMx, sp.BCJ, sp.BCx);
        return std::make_tuple(sp, B, L, l);
        });

    m.def("write_fast_cd_sim_static_precomputation", [](std::string cache_dir, fast_cd_arap_static_precomp& sp, EigenDRef<MatrixXd> B, const VectorXd& L, const VectorXi& l) {
        write_fast_cd_sim_static_precomputation(cache_dir, B, L, l, sp.BCB, sp.BMB, sp.BAB,
            sp.AeqB, sp.GmKB, sp.GmKJ, sp.GmKx, sp.G1VKB, sp.BMJ, sp.BMx, sp.BCJ, sp.BCx);
        });

    m.def("fit_rig_to_mesh", [](MatrixXd W, MatrixXd P0, 
        MatrixXd V0, MatrixXd X, MatrixXi T) {
            MatrixXd W2 = W, P2 = P0;
            fit_rig_to_mesh(W2, P2, V0, X, T);
            return std::make_tuple(W2, P2);
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
        

}