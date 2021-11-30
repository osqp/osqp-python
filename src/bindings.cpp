#include <pybind11/pybind11.h>
#include "ext/include/hello.h"
#include "osqp_api_functions.h"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(ext, m) {
    m.attr("__name__") = "osqp.ext";
    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    py::class_<Hello>(m, "Hello")
    .def(py::init<>())
    .def("greet", &Hello::greet)
    .def("version", &osqp_version);
}
