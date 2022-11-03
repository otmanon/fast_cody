#include "fast_cd_arap_sim.h"


#include "rig_parameters.h"
#include "get_skeleton_mesh.h"

#include "skinning_modes.h"
#include "get_modes.h"
#include "compute_clusters_igl.h"

#include "fast_cd_viewer.h"
#include "lbs_jacobian.h"


#include <igl/readMSH.h>
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
        .def(py::init<>());


    py::class_<fast_cd_sim_params>(m, "fast_cd_sim_params")
        .def(py::init<>())
        .def(py::init<EigenDRef<MatrixXd>, EigenDRef<MatrixXi>,
            EigenDRef<MatrixXd>, EigenDRef<VectorXi>,
            SparseMatrix<double>&, double, double, double,
            bool >());
       
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
                auto wrapperFunc = [=](igl::opengl::glfw::Viewer&) -> bool {  //pase by value here is reaaaly important. 
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

    m.def("lbs_jacobian", [](EigenDRef<MatrixXd> V, EigenDRef<MatrixXd> W) {
        SparseMatrix<double> J;
        lbs_jacobian(V, W, J);
        return J;
        });

        
}