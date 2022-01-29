import os
import shutil
import sys
from glob import glob
from platform import system
from shutil import copyfile, copy
from subprocess import call, check_output, check_call, CalledProcessError, STDOUT

from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext
import distutils.sysconfig as sysconfig


cmake_args = ["-DUNITTESTS=OFF"]
cmake_build_flags = []
define_macros = []
lib_subdir = []

if system() == 'Windows':
    cmake_args += ['-G', 'Visual Studio 14 2015']
    if sys.maxsize // 2 ** 32 > 0:
        cmake_args[-1] += ' Win64'
    cmake_build_flags += ['--config', 'Release']
    lib_name = 'osqp.lib'
    lib_subdir = ['Release']

else:
    cmake_args += ['-G', 'Unix Makefiles']
    lib_name = 'libosqp.a'

# Pass Python option to CMake and Python interface compilation
# Note: DDLONG=OFF due to
#   https://github.com/numpy/numpy/issues/5906
#   https://github.com/ContinuumIO/anaconda-issues/issues/3823
cmake_args += ['-DPYTHON=ON', '-DDLONG=OFF']

# Pass python to compiler launched from setup.py
define_macros += [('PYTHON', None)]

# Pass python include dirs to cmake
cmake_args += ['-DPYTHON_INCLUDE_DIRS=%s' % sysconfig.get_python_inc()]


# Define osqp and qdldl directories
current_dir = os.getcwd()
osqp_dir = os.path.join('osqp_sources')
osqp_ext_src_dir = os.path.join('src', 'extension', 'src')
osqp_build_dir = os.path.join(osqp_dir, 'build')
qdldl_dir = os.path.join(osqp_dir, 'lin_sys', 'direct', 'qdldl')


# Interface files
class get_numpy_include(object):
    """Returns Numpy's include path with lazy import.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


# Set optimizer flag
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []

# External libraries
library_dirs = []
libraries = []
if system() == 'Linux':
    libraries += ['rt']
if system() == 'Windows':
    # They moved the stdio library to another place.
    # We need to include this to fix the dependency
    libraries += ['legacy_stdio_definitions']


def prepare_codegen(osqp_dir, qdldl_dir):
    osqp_codegen_sources_dir = os.path.join('src', 'osqp', 'codegen', 'sources')
    if os.path.exists(osqp_codegen_sources_dir):
        shutil.rmtree(osqp_codegen_sources_dir)
    os.makedirs(osqp_codegen_sources_dir)

    # OSQP C files
    cfiles = [os.path.join(osqp_dir, 'src', f)
              for f in os.listdir(os.path.join(osqp_dir, 'src'))
              if f.endswith('.c') and f not in ('cs.c', 'ctrlc.c', 'polish.c',
                                                'lin_sys.c')]
    cfiles += [os.path.join(qdldl_dir, f)
               for f in os.listdir(qdldl_dir)
               if f.endswith('.c')]
    cfiles += [os.path.join(qdldl_dir, 'qdldl_sources', 'src', f)
               for f in os.listdir(os.path.join(qdldl_dir, 'qdldl_sources',
                                                'src'))]
    osqp_codegen_sources_c_dir = os.path.join(osqp_codegen_sources_dir, 'src')
    if os.path.exists(osqp_codegen_sources_c_dir):  # Create destination directory
        shutil.rmtree(osqp_codegen_sources_c_dir)
    os.makedirs(osqp_codegen_sources_c_dir)
    for f in cfiles:  # Copy C files
        copy(f, osqp_codegen_sources_c_dir)

    # List with OSQP H files
    hfiles = [os.path.join(osqp_dir, 'include', f)
              for f in os.listdir(os.path.join(osqp_dir, 'include'))
              if f.endswith('.h') and f not in ('qdldl_types.h',
                                                'osqp_configure.h',
                                                'cs.h', 'ctrlc.h', 'polish.h',
                                                'lin_sys.h')]
    hfiles += [os.path.join(qdldl_dir, f)
               for f in os.listdir(qdldl_dir)
               if f.endswith('.h')]
    hfiles += [os.path.join(qdldl_dir, 'qdldl_sources', 'include', f)
               for f in os.listdir(os.path.join(qdldl_dir, 'qdldl_sources',
                                                'include'))
               if f.endswith('.h')]
    osqp_codegen_sources_h_dir = os.path.join(osqp_codegen_sources_dir, 'include')
    if os.path.exists(osqp_codegen_sources_h_dir):  # Create destination directory
        shutil.rmtree(osqp_codegen_sources_h_dir)
    os.makedirs(osqp_codegen_sources_h_dir)
    for f in hfiles:  # Copy header files
        copy(f, osqp_codegen_sources_h_dir)

    # List with OSQP configure files
    configure_files = [os.path.join(osqp_dir, 'configure', 'osqp_configure.h.in'),
                       os.path.join(qdldl_dir, 'qdldl_sources', 'configure',
                                    'qdldl_types.h.in')]
    osqp_codegen_sources_configure_dir = os.path.join(osqp_codegen_sources_dir,
                                                      'configure')
    if os.path.exists(osqp_codegen_sources_configure_dir):
        shutil.rmtree(osqp_codegen_sources_configure_dir)
    os.makedirs(osqp_codegen_sources_configure_dir)
    for f in configure_files:  # Copy configure files
        copy(f, osqp_codegen_sources_configure_dir)

    # Copy cmake files
    copy(os.path.join(osqp_dir, 'src',     'CMakeLists.txt'),
         osqp_codegen_sources_c_dir)
    copy(os.path.join(osqp_dir, 'include', 'CMakeLists.txt'),
         osqp_codegen_sources_h_dir)

_osqp = Extension('osqp._osqp',
                  define_macros=define_macros,
                  libraries=libraries,
                  library_dirs=library_dirs,
                  include_dirs=[
                        os.path.join(osqp_dir, 'include'),      # osqp.h
                        os.path.join(qdldl_dir),                # qdldl_interface header to extract workspace for codegen
                        os.path.join(qdldl_dir, "qdldl_sources", "include"),     # qdldl includes for file types
                        os.path.join('src', 'extension', 'include'),   # auxiliary .h files
                        get_numpy_include()
                  ],
                  extra_objects=[os.path.join('src', 'extension', 'src', lib_name)],
                  sources=glob(os.path.join('src', 'extension', 'src', '*.c')),
                  extra_compile_args=compile_args)
_osqp.cmake_args = cmake_args

prepare_codegen(osqp_dir, qdldl_dir)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', cmake_args=None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.cmake_args = cmake_args


class CmdCMakeBuild(build_ext):
    def build_extension(self, ext):
        if ext.name == 'osqp._osqp':
            self.build_extension_legacy(ext, osqp_ext_src_dir, osqp_build_dir)
        else:
            self.build_extension_pybind11(ext)
        super().build_extension(ext)

    def build_extension_legacy(self, ext, src_dir, build_dir):
        # Compile OSQP using CMake

        # Create build directory
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        os.chdir(build_dir)

        try:
            check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build OSQP")

        # Compile static library with CMake
        call(['cmake'] + ext.cmake_args + ['..', '-DUNITTESTS=OFF'])
        call(['cmake', '--build', '.', '--target', 'osqpstatic'] +
             cmake_build_flags)

        # Change directory back to the python interface
        os.chdir(current_dir)

        # Copy static library to src folder
        lib_origin = [build_dir, 'out'] + lib_subdir + [lib_name]
        lib_origin = os.path.join(*lib_origin)
        copyfile(lib_origin, os.path.join(src_dir, lib_name))

    def build_extension_pybind11(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DBUILD_TESTING=OFF']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if system() == "Windows":
            cmake_args += ['-G', 'Visual Studio 16 2019']
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if os.path.exists(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp)

        _ext_name = ext.name.split('.')[-1]
        cmake_args.extend([f'-DOSQP_EXT_MODULE_NAME={_ext_name}'])

        # What variables from the environment do we wish to pass on to cmake as variables?
        cmake_env_vars = ('CMAKE_CUDA_COMPILER', 'CUDA_TOOLKIT_ROOT_DIR', 'MKL_DIR', 'MKL_ROOT')
        for cmake_env_var in cmake_env_vars:
            cmake_var = os.environ.get(cmake_env_var)
            if cmake_var:
                cmake_args.extend([f'-D{cmake_env_var}={cmake_var}'])

        if ext.cmake_args is not None:
            cmake_args.extend(ext.cmake_args)
            
        check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


algebra = os.environ.get('OSQP_ALGEBRA', 'default')
assert algebra in ('default', 'mkl', 'cuda'), f'Unknown algebra f{algebra}'
if algebra == 'default':
    package_name = 'osqpdev'
    ext_modules = [_osqp, CMakeExtension(f'osqp.ext_default', cmake_args=['-DALGEBRA=default'])]
else:
    package_name = f'osqp_{algebra}'
    ext_modules = [CMakeExtension(f'osqp_{algebra}', cmake_args=[f'-DALGEBRA={algebra}'])]

setup(name=package_name,
      author='Bartolomeo Stellato, Goran Banjac',
      author_email='bartolomeo.stellato@gmail.com',
      description='OSQP: The Operator Splitting QP Solver',
      long_description=open('README.rst').read(),
      package_dir={'': 'src'},
      include_package_data=True,
      install_requires=['numpy>=1.7', 'scipy>=0.13.2', 'qdldl'],
      python_requires='>=3.7',
      license='Apache 2.0',
      url="https://osqp.org/",
      cmdclass={'build_ext': CmdCMakeBuild},
      packages=find_namespace_packages(where='src'),
      ext_modules=ext_modules
)
