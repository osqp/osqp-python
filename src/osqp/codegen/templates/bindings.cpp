#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "osqp_api_functions.h"
#include "osqp_api_types.h"
#include "workspace.h"

py::tuple solve() {
    py::gil_scoped_release release;
    c_int status = osqp_solve(&solver);
    py::gil_scoped_acquire acquire;

    if (status != 0) throw std::runtime_error("Solve failed");

    auto x = py::array_t<c_float>({2}, {sizeof(c_float)}, (&solver)->solution->x);
    auto y = py::array_t<c_float>({5}, {sizeof(c_float)}, (&solver)->solution->y);

    py::tuple results = py::make_tuple(x, y, status, (&solver)->info->iter, (&solver)->info->run_time);
    return results;
}

c_int update_data_vec(py::object q, py::object l, py::object u) {
    c_float* _q;
    c_float* _l;
    c_float* _u;

    if (q.is_none()) {
        _q = NULL;
    } else {
        _q = (c_float *)py::array_t<c_float>(q).data();
    }
    if (l.is_none()) {
        _l = NULL;
    } else {
        _l = (c_float *)py::array_t<c_float>(l).data();
    }
    if (u.is_none()) {
        _u = NULL;
    } else {
        _u = (c_float *)py::array_t<c_float>(u).data();
    }

    return osqp_update_data_vec(&solver, _q, _l, _u);
}

PYBIND11_MODULE(foo, m) {
    m.def("solve", &solve);
    m.def("update_data_vec", &update_data_vec, "Update q/l/u", py::arg("q") = py::none(), py::arg("l") = py::none(), py::arg("u") = py::none());
}
