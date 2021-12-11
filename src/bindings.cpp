#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include "osqp_api_functions.h"
#include "osqp_api_types.h"

class CSC {
    public:
        CSC(py::object A);
        ~CSC();
        csc* getcsc();
    private:
        csc* _csc;
        py::array_t<c_int> _p;
        py::array_t<c_int> _i;
        py::array_t<c_float> _x;
};

CSC::CSC(py::object A) {
    py::object spa = py::module::import("scipy.sparse");

    py::tuple dim = A.attr("shape");
    int m = dim[0].cast<int>();
    int n = dim[1].cast<int>();

    if (!spa.attr("isspmatrix_csc")(A)) A = spa.attr("csc_matrix")(A);

    this->_p = A.attr("indptr").cast<py::array_t<c_int, py::array::c_style>>();
    this->_i = A.attr("indices").cast<py::array_t<c_int, py::array::c_style>>();
    this->_x = A.attr("data").cast<py::array_t<c_float, py::array::c_style>>();

    this->_csc = new csc();
    this->_csc->m = m;
    this->_csc->n = n;
    this->_csc->p = (c_int *)this->_p.data();
    this->_csc->i = (c_int *)this->_i.data();
    this->_csc->x = (c_float *)this->_x.data();
    this->_csc->nzmax = A.attr("nnz").cast<int>();
    this->_csc->nz = -1;
}

csc *CSC::getcsc() {
    return this->_csc;
}

CSC::~CSC() {
    delete this->_csc;
}

OSQPSettings* init_OSQPSettings() {
    OSQPSettings* settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
    if (settings) {
        osqp_set_default_settings(settings);
    }
    return settings;
}

class MyOSQPSolver {
    public:
        MyOSQPSolver(CSC&, py::array_t<c_float>, CSC&, py::array_t<c_float>, py::array_t<c_float>, c_int, c_int, const OSQPSettings*);
        ~MyOSQPSolver();
        OSQPInfo* solve();
    private:
        OSQPSolver *_solver;
};

MyOSQPSolver::MyOSQPSolver(
        CSC& P,
        const py::array_t<c_float> q,
        CSC& A,
        const py::array_t<c_float> l,
        const py::array_t<c_float> u,
        c_int m,
        c_int n,
        const OSQPSettings *settings
) {
    this->_solver = new OSQPSolver();
    const c_float* _q = q.data();
    const c_float* _l = l.data();
    const c_float* _u = u.data();
    osqp_setup(&this->_solver, P.getcsc(), _q, A.getcsc(), _l, _u, m, n, settings);
}

MyOSQPSolver::~MyOSQPSolver() {
    delete this->_solver;
}

OSQPInfo* MyOSQPSolver::solve() {
    osqp_solve(this->_solver);
    return this->_solver->info;
}

PYBIND11_MODULE(ext, m) {
    m.attr("__name__") = "osqp.ext";

    py::class_<CSC>(m, "CSC")
    .def(py::init<py::object>());

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

    py::class_<MyOSQPSolver>(m, "OSQPSolver")
    .def(py::init<CSC&, const py::array_t<c_float>, CSC&, const py::array_t<c_float>, const py::array_t<c_float>, c_int, c_int, const OSQPSettings*>(),
            "P"_a, "q"_a.noconvert(), "A"_a, "l"_a.noconvert(), "u"_a.noconvert(), "m"_a, "n"_a, "settings"_a)
    .def("solve", &MyOSQPSolver::solve);
}
