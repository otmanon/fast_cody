#include <igl/readMSH.h>
#include <igl/massmatrix.h>
#include <igl/readOBJ.h>
#include <igl/writeDMAT.h>
#include <igl/boundary_facets.h>


#include <Eigen/Core>
#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
using namespace Eigen;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;
using namespace std;

void bind_igl(py::module& m) {

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
    return std::make_tuple(F, FiT, K);
        });

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

    m.def("readOBJ_tex", [](std::string filename) {
        MatrixXd V; MatrixXi F;
    MatrixXd TC; MatrixXi FTC;
    MatrixXd N; MatrixXi FN;
    igl::readOBJ(filename, V,TC, N,  F, FTC, FN);
    return std::make_tuple(V, TC, N, F, FTC, FN);
        }, "\
    Inputs : \n \
      filename (string) - file path of texture png\n \
    Outputs: \n \
      V - (n x 3) \n \
      TC - (n x 2) \n \
      N - (n x 3) \n \
      F - (F x 3) \n \
      FTC - (F x 2) \n \
      FN - (F x 3) \n \
    ");

}
