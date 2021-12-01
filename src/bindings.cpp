#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "osqp_api_functions.h"
#include "osqp_api_types.h"

#include "ext/include/hello.h"


OSQPSettings* init_OSQPSettings() {
  OSQPSettings* settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
  if (settings) {
    osqp_set_default_settings(settings);
  }
  return settings;
}


PYBIND11_MODULE(ext, m) {
    m.attr("__name__") = "osqp.ext";

    py::class_<Hello>(m, "Hello")
    .def(py::init<>())
    .def("greet", &Hello::greet)
    .def("version", &osqp_version);

    py::class_<OSQPSettings>(m, "OSQPSettings")
    .def(py::init(&init_OSQPSettings))
    .def_readwrite("rho", &OSQPSettings::rho);
}
