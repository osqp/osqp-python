import osqp
import numpy as np
from scipy import sparse


if __name__ == '__main__':
    # Define problem data
    P = sparse.csc_matrix([[4, 1], [1, 2]])
    q = np.array([1, 1])
    A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
    l = np.array([1, 0, 0])
    u = np.array([1, 0.7, 0.7])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1.0)

    # The OSQP object has "capabilities" that define what it can do.
    assert prob.has_capability('OSQP_CAPABILITY_CODEGEN')

    # Generate C code
    # fmt: off
    prob.codegen(
        'out',                     # Output folder for auto-generated code
        prefix='prob1_',           # Prefix for filenames and C variables; useful if generating multiple problems
        force_rewrite=True,        # Force rewrite if output folder exists?
        parameters='vectors',      # What do we wish to update in the generated code?
                                   # One of 'vectors' (allowing update of q/l/u through prob.update_data_vec)
                                   # or 'matrices' (allowing update of P/A/q/l/u
                                   # through prob.update_data_vec or prob.update_data_mat)
        use_float=False,           # Use single precision in generated code?
        printing_enable=False,     # Enable solver printing?
        profiling_enable=False,    # Enable solver profiling?
        interrupt_enable=False,    # Enable user interrupt (Ctrl-C)?
        include_codegen_src=True,  # Include headers/sources/Makefile in the output folder,
                                   # creating a self-contained compilable folder?
        extension_name='pyosqp',   # Name of the generated python extension; generates a setup.py; Set None to skip
        compile=False,             # Compile the above python extension into an importable module
                                   # (allowing "import pyosqp")?
    )
    # fmt: on
