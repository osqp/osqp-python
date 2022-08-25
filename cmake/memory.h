// Define memory allocation for python. Note that in Python 2 memory manager
// Calloc is not implemented
#   include <Python.h>
#   if PY_MAJOR_VERSION >= 3
// https://docs.python.org/3/c-api/memory.html
// The following function sets are wrappers to the system allocator. These functions are thread-safe, the GIL does not need to be held.
// The default raw memory allocator uses the following functions: malloc(), calloc(), realloc() and free(); call malloc(1) (or calloc(1, 1)) when requesting zero bytes.
#    define c_malloc PyMem_RawMalloc
#    define c_calloc PyMem_RawCalloc
#    define c_free PyMem_RawFree
#    define c_realloc PyMem_RawRealloc
#   else  /* if PY_MAJOR_VERSION >= 3 */
#   define c_malloc PyMem_Malloc
#   define c_free PyMem_Free
#   define c_realloc PyMem_Realloc
static void* c_calloc(size_t num, size_t size) {
    void *m = PyMem_Malloc(num * size);
    memset(m, 0, num * size);
    return m;
}
#   endif /* if PY_MAJOR_VERSION >= 3 */
