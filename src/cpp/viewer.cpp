#include "fast_cd_viewer.h"
#include "fast_cd_viewer_vertex_selector.h"
#include "fast_cd_viewer_custom_shader.h"
#include "fast_cd_viewer_parameters.h"

#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;
using namespace std;

void bind_viewer(py::module& m) {

    /*
    Binding fast_cd_viewer
    */
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
         .def("set_key_callback", [&](fast_cd_viewer& v, std::function<bool(unsigned int, int)>& func)
            {
                auto wrapperFunc = [=](igl::opengl::glfw::Viewer&, unsigned int key, int modifier) -> bool {
                return func(key, modifier);
                };
                v.igl_v->callback_key_pressed= wrapperFunc;
            })
        .def("set_face_based", &fast_cd_viewer::set_face_based)
        .def("set_color", static_cast<void (fast_cd_viewer::*)(const RowVector3d&, int)>(&fast_cd_viewer::set_color))
         .def("set_show_lines", &fast_cd_viewer::get_show_lines)
        .def("set_show_lines", &fast_cd_viewer::set_show_lines)
        .def("set_show_faces", &fast_cd_viewer::set_show_faces)
        .def("get_show_faces", &fast_cd_viewer::get_show_faces)
        .def("launch", &fast_cd_viewer::launch)
        .def("init_guizmo", [&](fast_cd_viewer& v, bool visible, EigenDRef<Matrix4f> A0, std::function<void(const Matrix4f&)> func, std::string operation)
            {
            v.guizmo->visible = visible;
            v.guizmo->T = A0;
            auto wrapperFunc = [=](const Matrix4f& A) {
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
         })
         .def("set_texture", &fast_cd_viewer::set_texture, 
             " \
             const string & tex_png, const  MatrixXd & TC, const  MatrixXi & FTC, int id \
             ");
                    // .def("set_points", &fast_cd_viewer::set_points)

    /*
    Binding fast_cd_viewer_vertex_selector class that allows easy vertex
    selection
    */



py::class_<fast_cd_viewer_vertex_selector,
fast_cd_viewer>(m, "fast_cd_viewer_vertex_selector")
    .def(py::init<>())
    .def("query_new_handles",
        [](fast_cd_viewer_vertex_selector& v) {
            MatrixXd C; VectorXi CI;
            bool new_handle = v.query_new_handles(C, CI);
            return std::make_tuple(C, CI, new_handle);
    }, "\
    Queries new handle information.Once this is called, \
    the vertex_added flag is set to false \n \ \
    Outputs: \n \
    C - n x 3 vertex handle positions \n \
    CI - n x 1 vertex handle indices \n \
    vertex_added - (bool)whether or not\ \n \
    a new vertex was added this timestep \n \
    ")
    .def("query_new_handles_on_mesh",
    [](fast_cd_viewer_vertex_selector& v, 
        EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> T) {
        MatrixXd C; VectorXi CI;
        bool new_handle = v.query_new_handles_on_mesh(C, CI, V, T);
        return std::make_tuple(C, CI, new_handle);
    }, "\
    Queries new handle information and \
     if a new handle was added this timestep,\
    projects it to mesh V, T \
    Once this is called, \
    the vertex_added flag is set to false \n \ \
    Outputs: \n \
    C - n x 3 vertex handle positions \n \
    CI - n x 1 vertex handle indices \n \
    vertex_added - (bool)whether or not\ \n \
    a new vertex was added this timestep \n \
    ");
       // .def("set_mesh", &fast_cd_viewer_vertex_selector::set_mesh);

    py::class_<fast_cd_viewer_custom_shader,
        fast_cd_viewer>(m, "fast_cd_viewer_custom_shader")
        .def(py::init<>())
        .def(py::init<string&, string&, int, int>(), py::arg("vertex_shader"), py::arg("framgnet_shader"),
            py::arg("max_num_primary_bones"), py::arg("max_num_secondary_bones"))
        .def("launch", &fast_cd_viewer_custom_shader::launch, py::arg("max_fps"), py::arg("render"))
        .def("init_buffers", &fast_cd_viewer_custom_shader::init_buffers)
        .def("free_buffers", &fast_cd_viewer_custom_shader::free_buffers)
        .def("set_primary_weights", &fast_cd_viewer_custom_shader::set_primary_weights)
        .def("set_secondary_weights", &fast_cd_viewer_custom_shader::set_secondary_weights)
        .def("set_weights", &fast_cd_viewer_custom_shader::set_weights)
        .def("set_primary_bone_transforms", &fast_cd_viewer_custom_shader::set_primary_bone_transforms)
        .def("set_secondary_bone_transforms", &fast_cd_viewer_custom_shader::set_secondary_bone_transforms)
        .def("set_bone_transforms", &fast_cd_viewer_custom_shader::set_bone_transforms)
        .def("updateGL", &fast_cd_viewer_custom_shader::updateGL);

    py::class_<fast_cd_viewer_parameters>(m, "fast_cd_viewer_parameters")
        .def(py::init<>())
        .def("set_texture", static_cast<void (fast_cd_viewer_parameters::*)
            (std::string, std::string, double, RowVector3d)>(&fast_cd_viewer_parameters::set_texture))
        .def("set_texture", static_cast<void (fast_cd_viewer_parameters::*)
            (std::string, std::string)>(&fast_cd_viewer_parameters::set_texture))
        ;
       
} 