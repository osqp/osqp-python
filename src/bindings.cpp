#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

using namespace pybind11::literals;

#include "osqp_api_functions.h"
#include "osqp_api_types.h"
#include "osqp_api_utils.h"

#include "ext/include/hello.h"


OSQPSettings* init_OSQPSettings() {
  OSQPSettings* settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
  if (settings) {
    osqp_set_default_settings(settings);
  }
  return settings;
}

OSQPSolver* init_OSQPSolver() {
  OSQPSolver* solver = (OSQPSolver *)malloc(sizeof(OSQPSolver));
  return solver;
}

c_int do_it(OSQPSolver* solver, csc* P, c_float* q, csc* A, c_float* l, c_float* u, c_int m, c_int n, OSQPSettings* settings) {
    c_int exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);
    return exitflag;
}


PYBIND11_MODULE(ext, m) {
    m.attr("__name__") = "osqp.ext";

    py::class_<Hello>(m, "Hello")
    .def(py::init<>())
    .def("greet", &Hello::greet)
    .def("version", &osqp_version);

    py::class_<csc>(m, "CSC");
    m.def("mysum", [](py::array_t<c_float> x) {
        auto r = x.unchecked<1>();
        const c_float* A = x.data();
        c_float sum = 0;
        py::ssize_t n = r.shape(0);
        for (int i=0; i<n; i++) {
            sum += A[i];
        }
        return sum;
    });

    py::class_<OSQPSettings>(m, "OSQPSettings")
    .def(py::init(&init_OSQPSettings))
    .def_readwrite("rho", &OSQPSettings::rho);

    py::class_<OSQPSolution>(m, "OSQPSolution")
    .def_readonly("x", &OSQPSolution::x)
    .def_readonly("y", &OSQPSolution::y)
    .def_readonly("prim_inf_cert", &OSQPSolution::prim_inf_cert)
    .def_readonly("dual_inf_cert", &OSQPSolution::dual_inf_cert);

    py::class_<OSQPInfo>(m, "OSQPInfo")
    .def_readonly("status", &OSQPInfo::status)
    .def_readonly("obj_val", &OSQPInfo::obj_val);

    py::class_<OSQPSolver>(m, "OSQPSolver")
    .def(py::init(&init_OSQPSolver));

    m.def("do_it", &do_it);


}
