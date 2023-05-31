This folder contains custom printing/memory management routines for the Python wrapper for OSQP.

During build time for osqp, the following options are passed on to cmake by setup.py:
```
cmake -DOSQP_CUSTOM_PRINTING=/path/to/printing.h -DOSQP_CUSTOM_MEMORY=/path/to/memory.h
```
