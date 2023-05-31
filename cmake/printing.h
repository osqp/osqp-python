# include <Python.h>
# define c_print(...)                              \
  {                                                  \
    PyGILState_STATE gilstate = PyGILState_Ensure(); \
    PySys_WriteStdout(__VA_ARGS__);                  \
    PyGILState_Release(gilstate);                    \
  }
