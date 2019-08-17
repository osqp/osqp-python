#ifndef OSQPMODULEMETHODS_H
#define OSQPMODULEMETHODS_H

/***********************************************************************
 * OSQP methods independently from any object                          *
 ***********************************************************************/

static PyObject * OSQP_module_solve(OSQP *self, PyObject *args, PyObject *kwargs) {
  c_int n, m;  // Problem dimensions
  c_int exitflag_setup, exitflag_solve;

  // Variables for setup
  PyOSQPData *pydata;
  OSQPData * data;
  OSQPSettings * settings;
  OSQPWorkspace * workspace;  // Pointer to C workspace structure
  PyArrayObject *Px, *Pi, *Pp, *q, *Ax, *Ai, *Ap, *l, *u;


  // Variables for solve
  // Create status object
  PyObject * status;
  // Create obj_val object
  PyObject * obj_val;
  // Create solution objects
  PyObject * x, *y, *prim_inf_cert, *dual_inf_cert;
  // Define info related variables
  static char *argparse_string_info;
  PyObject *info_list;
  PyObject *info;

  // Results
  PyObject *results_list;
  PyObject *results;

  npy_intp nd[1];
  npy_intp md[1];

  static char *kwlist[] = {"dims",                     // nvars and ncons
			   "Px", "Pi", "Pp", "q",      // Cost function
			   "Ax", "Ai", "Ap", "l", "u", // Constraints
			   "scaling",
			   "adaptive_rho", "adaptive_rho_interval",
			   "adaptive_rho_tolerance", "adaptive_rho_fraction",
			   "rho", "sigma", "max_iter", "eps_abs", "eps_rel",
			   "eps_prim_inf", "eps_dual_inf", "alpha", "delta",
			   "linsys_solver", "polish",
			   "polish_refine_iter", "verbose",
			   "scaled_termination",
			   "check_termination", "warm_start",
			   "time_limit", NULL};        // Settings

#ifdef DLONG

  // NB: linsys_solver is enum type which is stored as int (regardless on how c_int is defined).
#ifdef DFLOAT
  static char * argparse_string_setup = "(LL)O!O!O!O!O!O!O!O!O!|LLLffffLffffffiLLLLLLf";
#else
  static char * argparse_string_setup = "(LL)O!O!O!O!O!O!O!O!O!|LLLddddLddddddiLLLLLLd";
#endif

#else

#ifdef DFLOAT
  static char * argparse_string_setup = "(ii)O!O!O!O!O!O!O!O!O!|iiiffffiffffffiiiiiiif";
#else
  static char * argparse_string_setup = "(ii)O!O!O!O!O!O!O!O!O!|iiiddddiddddddiiiiiiid";
#endif

#endif

  // Initialize settings
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);

  if( !PyArg_ParseTupleAndKeywords(args, kwargs, argparse_string_setup, kwlist,
				   &n, &m,
				   &PyArray_Type, &Px,
				   &PyArray_Type, &Pi,
				   &PyArray_Type, &Pp,
				   &PyArray_Type, &q,
				   &PyArray_Type, &Ax,
				   &PyArray_Type, &Ai,
				   &PyArray_Type, &Ap,
				   &PyArray_Type, &l,
				   &PyArray_Type, &u,
				   &settings->scaling,
				   &settings->adaptive_rho,
				   &settings->adaptive_rho_interval,
				   &settings->adaptive_rho_tolerance,
				   &settings->adaptive_rho_fraction,
				   &settings->rho,
				   &settings->sigma,
				   &settings->max_iter,
				   &settings->eps_abs,
				   &settings->eps_rel,
				   &settings->eps_prim_inf,
				   &settings->eps_dual_inf,
				   &settings->alpha,
				   &settings->delta,
				   &settings->linsys_solver,
				   &settings->polish,
				   &settings->polish_refine_iter,
				   &settings->verbose,
				   &settings->scaled_termination,
				   &settings->check_termination,
				   &settings->warm_start,
				   &settings->time_limit)) {
    return (PyObject *) NULL;
  }

  // Create Data from parsed vectors
  pydata = create_pydata(n, m, Px, Pi, Pp, q, Ax, Ai, Ap, l, u);
  data = create_data(pydata);

  // Perform setup and solve
  // release the GIL
  Py_BEGIN_ALLOW_THREADS;
  // Create Workspace object
  exitflag_setup = osqp_setup(&(workspace), data, settings);
  exitflag_solve = osqp_solve(workspace);
  // reacquire the GIL
  Py_END_ALLOW_THREADS;
  
  // Cleanup data and settings
  free_data(data, pydata);
  c_free(settings);

  // Check successful setup and solve
  if (exitflag_setup){ // Workspace allocation error
    PyErr_SetString(PyExc_ValueError, "Workspace allocation error!");
    return (PyObject *) NULL;
  }

  if(exitflag_solve){
      PyErr_SetString(PyExc_ValueError, "OSQP solve error!");
      return (PyObject *) NULL;
  }

  // Temporary solution
  nd[0] = (npy_intp)workspace->data->n;  // Dimensions in R^n
  md[0] = (npy_intp)workspace->data->m;  // Dimensions in R^m

  // If problem is not primal or dual infeasible store it
  if ((workspace->info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (workspace->info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (workspace->info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (workspace->info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE)){

    // Primal and dual solutions
    x = (PyObject *)PyArrayFromCArray(workspace->solution->x, nd);
    y = (PyObject *)PyArrayFromCArray(workspace->solution->y, md);

    // Infeasibility certificates -> None values
    prim_inf_cert = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
    dual_inf_cert = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

  } else if (workspace->info->status_val == OSQP_PRIMAL_INFEASIBLE ||
	     workspace->info->status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE) {
    // primal infeasible

    // Primal and dual solution arrays -> None values
    x = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
    y = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

    // Primal infeasibility certificate
    prim_inf_cert = (PyObject *)PyArrayFromCArray(workspace->delta_y, md);

    // Dual infeasibility certificate -> None values
    dual_inf_cert = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);

    // Set objective value to infinity
    workspace->info->obj_val = NPY_INFINITY;

  } else {
    // dual infeasible

    // Primal and dual solution arrays -> None values
    x = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
    y = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

    // Primal infeasibility certificate -> None values
    prim_inf_cert = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

    // Dual infeasibility certificate
    dual_inf_cert = (PyObject *)PyArrayFromCArray(workspace->delta_x, nd);

    // Set objective value to -infinity
    workspace->info->obj_val = -NPY_INFINITY;
  }

  /*  CREATE INFO OBJECT */
  // Store status string
  status = PyUnicode_FromString(workspace->info->status);

  // Store obj_val
  if (workspace->info->status_val == OSQP_NON_CVX) {	// non convex
    obj_val = PyFloat_FromDouble(Py_NAN);
  } else {
    obj_val = PyFloat_FromDouble(workspace->info->obj_val);
  }

#ifdef PROFILING

#ifdef DLONG

#ifdef DFLOAT
  argparse_string_info = "LOLLOfffffffLf";
#else
  argparse_string_info = "LOLLOdddddddLd";
#endif

#else

#ifdef DFLOAT
  argparse_string_info = "iOiiOfffffffif";
#else
  argparse_string_info = "iOiiOdddddddid";
#endif

#endif /* DLONG */

  info_list = Py_BuildValue(argparse_string_info,
			    workspace->info->iter,
			    status,
			    workspace->info->status_val,
			    workspace->info->status_polish,
			    obj_val,
			    workspace->info->pri_res,
			    workspace->info->dua_res,
			    workspace->info->setup_time,
			    workspace->info->solve_time,
			    workspace->info->update_time,
			    workspace->info->polish_time,
			    workspace->info->run_time,
			    workspace->info->rho_updates,
			    workspace->info->rho_estimate
			    );
#else /* PROFILING */

#ifdef DLONG

#ifdef DFLOAT
  argparse_string = "LOLLOffLf";
#else
  argparse_string = "LOLLOddLd";
#endif

#else

#ifdef DFLOAT
  argparse_string = "iOiiOffif";
#else
  argparse_string = "iOiiOddid";
#endif

#endif /* DLONG */

  info_list = Py_BuildValue(argparse_string_info,
			    workspace->info->iter,
			    status,
			    workspace->info->status_val,
			    workspace->info->status_polish,
			    obj_val,
			    workspace->info->pri_res,
			    workspace->info->dua_res,
			    workspace->info->rho_updates,
			    workspace->info->rho_estimate,
			    );
#endif /* PROFILING */

  info = PyObject_CallObject((PyObject *) &OSQP_info_Type, info_list);

  /* Release the info argument list. */
  Py_DECREF(info_list);

  /*  CREATE RESULTS OBJECT */
  results_list = Py_BuildValue("OOOOO", x, y, prim_inf_cert, dual_inf_cert, info);

  /* Call the class object. */
  results = PyObject_CallObject((PyObject *) &OSQP_results_Type, results_list);

  // Delete results list
  Py_DECREF(results_list);

  // Cleanup workspace
  if (osqp_cleanup(workspace)) {
    PyErr_SetString(PyExc_ValueError, "Workspace deallocation error!");
    return (PyObject *) NULL;
  }

  // Return results    
  return results;


}


static PyObject *OSQP_constant(OSQP *self, PyObject *args) {

    char * constant_name;  // String less than 32 chars

    // Parse arguments
    if( !PyArg_ParseTuple(args, "s", &(constant_name))) {
        return (PyObject *) NULL;
    }


    if(!strcmp(constant_name, "OSQP_INFTY")){
#ifdef DFLOAT
        return Py_BuildValue("f", OSQP_INFTY);
#else
        return Py_BuildValue("d", OSQP_INFTY);
#endif
    }

    if(!strcmp(constant_name, "OSQP_NAN")){
#ifdef DFLOAT
        return Py_BuildValue("f", Py_NAN);
#else
        return Py_BuildValue("d", Py_NAN);
#endif
    }

    if(!strcmp(constant_name, "OSQP_SOLVED")){
        return Py_BuildValue("i", OSQP_SOLVED);
    }

    if(!strcmp(constant_name, "OSQP_SOLVED_INACCURATE")){
        return Py_BuildValue("i", OSQP_SOLVED_INACCURATE);
    }

    if(!strcmp(constant_name, "OSQP_UNSOLVED")){
        return Py_BuildValue("i", OSQP_UNSOLVED);
    }

    if(!strcmp(constant_name, "OSQP_PRIMAL_INFEASIBLE")){
        return Py_BuildValue("i", OSQP_PRIMAL_INFEASIBLE);
    }

	if(!strcmp(constant_name, "OSQP_PRIMAL_INFEASIBLE_INACCURATE")){
		return Py_BuildValue("i", OSQP_PRIMAL_INFEASIBLE_INACCURATE);
	}

    if(!strcmp(constant_name, "OSQP_DUAL_INFEASIBLE")){
        return Py_BuildValue("i", OSQP_DUAL_INFEASIBLE);
    }

	if(!strcmp(constant_name, "OSQP_DUAL_INFEASIBLE_INACCURATE")){
		return Py_BuildValue("i", OSQP_DUAL_INFEASIBLE_INACCURATE);
	}

    if(!strcmp(constant_name, "OSQP_MAX_ITER_REACHED")){
        return Py_BuildValue("i", OSQP_MAX_ITER_REACHED);
    }

    if(!strcmp(constant_name, "OSQP_NON_CVX")){
        return Py_BuildValue("i", OSQP_NON_CVX);
    }

    if(!strcmp(constant_name, "OSQP_TIME_LIMIT_REACHED")){
        return Py_BuildValue("i", OSQP_TIME_LIMIT_REACHED);
    }

	// Linear system solvers
	if(!strcmp(constant_name, "QDLDL_SOLVER")){
		return Py_BuildValue("i", QDLDL_SOLVER);
	}

	if(!strcmp(constant_name, "MKL_PARDISO_SOLVER")){
		return Py_BuildValue("i", MKL_PARDISO_SOLVER);
	}

    // If reached here error
    PyErr_SetString(PyExc_ValueError, "Constant not recognized");
    return (PyObject *) NULL;
}




static PyMethodDef OSQP_module_methods[] = {
					    {"solve", (PyCFunction)OSQP_module_solve,METH_VARARGS|METH_KEYWORDS, PyDoc_STR("Setup solve and cleanup OSQP problem. This function releases GIL.")},
					        {"constant", (PyCFunction)OSQP_constant, METH_VARARGS, PyDoc_STR("Return internal OSQP constant")},
					    {NULL, NULL}		/* sentinel */
};

  
#endif
