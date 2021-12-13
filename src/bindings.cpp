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
        py::tuple solve();
        c_int update_data_vec(py::object, py::object, py::object);
        c_int update_settings(const OSQPSettings&);
        OSQPSettings* get_settings();
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

OSQPSettings* MyOSQPSolver::get_settings() {
    return this->_solver->settings;
}

py::tuple MyOSQPSolver::solve() {
    osqp_solve(this->_solver);
    return py::make_tuple(this->_solver->solution, this->_solver->info);
}

c_int MyOSQPSolver::update_settings(const OSQPSettings& new_settings) {
    return osqp_update_settings(this->_solver, &new_settings);
}

c_int MyOSQPSolver::update_data_vec(py::object q, py::object l, py::object u) {
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

    return osqp_update_data_vec(this->_solver, _q, _l, _u);
}

PYBIND11_MODULE(ext, m) {
    m.attr("__name__") = "osqp.ext";

    py::class_<CSC>(m, "CSC")
    .def(py::init<py::object>());

    py::enum_<linsys_solver_type>(m, "linsys_solver_type")
    .value("DIRECT_SOLVER", DIRECT_SOLVER)
    .value("INDIRECT_SOLVER", INDIRECT_SOLVER)
    .export_values();

    py::enum_<osqp_status_type>(m, "osqp_status_type")
    .value("OSQP_MAX_ITER_REACHED", OSQP_MAX_ITER_REACHED)
    .export_values();

    py::class_<OSQPSettings>(m, "OSQPSettings")
    .def(py::init(&init_OSQPSettings))

    .def_readwrite("device", &OSQPSettings::device)
    .def_readwrite("linsys_solver", &OSQPSettings::linsys_solver)
    .def_readwrite("verbose", &OSQPSettings::verbose)
    .def_readwrite("warm_starting", &OSQPSettings::warm_starting)
    .def_readwrite("scaling", &OSQPSettings::scaling)
    .def_readwrite("polishing", &OSQPSettings::polishing)

    // ADMM
    .def_readwrite("rho", &OSQPSettings::rho)
    .def_readwrite("rho_is_vec", &OSQPSettings::rho_is_vec)
    .def_readwrite("sigma", &OSQPSettings::sigma)
    .def_readwrite("alpha", &OSQPSettings::alpha)

    // CG
    .def_readwrite("cg_max_iter", &OSQPSettings::cg_max_iter)
    .def_readwrite("cg_tol_reduction", &OSQPSettings::cg_tol_reduction)
    .def_readwrite("cg_tol_fraction", &OSQPSettings::cg_tol_fraction)

    // Adaptive rho
    .def_readwrite("adaptive_rho", &OSQPSettings::adaptive_rho)
    .def_readwrite("adaptive_rho_interval", &OSQPSettings::adaptive_rho_interval)
    .def_readwrite("adaptive_rho_fraction", &OSQPSettings::adaptive_rho_fraction)
    .def_readwrite("adaptive_rho_tolerance", &OSQPSettings::adaptive_rho_tolerance)

    // Termination parameters
    .def_readwrite("max_iter", &OSQPSettings::max_iter)
    .def_readwrite("eps_abs", &OSQPSettings::eps_abs)
    .def_readwrite("eps_rel", &OSQPSettings::eps_rel)
    .def_readwrite("eps_prim_inf", &OSQPSettings::eps_prim_inf)
    .def_readwrite("eps_dual_inf", &OSQPSettings::eps_dual_inf)
    .def_readwrite("scaled_termination", &OSQPSettings::scaled_termination)
    .def_readwrite("check_termination", &OSQPSettings::check_termination)
    .def_readwrite("time_limit", &OSQPSettings::time_limit)

    // Polishing
    .def_readwrite("delta", &OSQPSettings::delta)
    .def_readwrite("polish_refine_iter", &OSQPSettings::polish_refine_iter);

    py::class_<OSQPSolution>(m, "OSQPSolution")
    .def_readonly("x", &OSQPSolution::x)
    .def_readonly("y", &OSQPSolution::y)
    .def_readonly("prim_inf_cert", &OSQPSolution::prim_inf_cert)
    .def_readonly("dual_inf_cert", &OSQPSolution::dual_inf_cert);

    py::class_<OSQPInfo>(m, "OSQPInfo")
    .def_readonly("status", &OSQPInfo::status)
    .def_readonly("status_val", &OSQPInfo::status_val)
    .def_readonly("status_polish", &OSQPInfo::status_polish)
    .def_readonly("obj_val", &OSQPInfo::obj_val)
    .def_readonly("prim_res", &OSQPInfo::prim_res)
    .def_readonly("dual_res", &OSQPInfo::dual_res)
    .def_readonly("iter", &OSQPInfo::iter)
    .def_readonly("rho_updates", &OSQPInfo::rho_updates)
    .def_readonly("rho_estimate", &OSQPInfo::rho_estimate)
    .def_readonly("setup_time", &OSQPInfo::setup_time)
    .def_readonly("solve_time", &OSQPInfo::solve_time)
    .def_readonly("update_time", &OSQPInfo::update_time)
    .def_readonly("polish_time", &OSQPInfo::polish_time)
    .def_readonly("run_time", &OSQPInfo::run_time);

    py::class_<MyOSQPSolver>(m, "OSQPSolver")
    .def(py::init<CSC&, const py::array_t<c_float>, CSC&, const py::array_t<c_float>, const py::array_t<c_float>, c_int, c_int, const OSQPSettings*>(),
            "P"_a, "q"_a.noconvert(), "A"_a, "l"_a.noconvert(), "u"_a.noconvert(), "m"_a, "n"_a, "settings"_a)
    .def("solve", &MyOSQPSolver::solve)
    .def("update_data_vec", &MyOSQPSolver::update_data_vec, "q"_a.none(true), "l"_a.none(true), "u"_a.none(true))
    .def("update_settings", &MyOSQPSolver::update_settings)
    .def("get_settings", &MyOSQPSolver::get_settings, py::return_value_policy::reference);
}