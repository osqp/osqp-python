import os
import shutil
import sys
from glob import glob
from platform import system
from subprocess import check_call

from distutils.sysconfig import get_python_inc
from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', cmake_args=None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.cmake_args = cmake_args


class CustomBuildPy(build_py):
    def run(self):
        # Build all extensions first so that we generate codegen files in the build folder
        # Note that each command like 'build_ext', is run once by setuptools, even if invoked multiple times.
        self.run_command('build_ext')

        codegen_build_dir = None
        for data_file in self.data_files:
            package, src_dir, build_dir, filename = data_file
            if package == 'osqp.codegen':
                codegen_build_dir = build_dir

        if codegen_build_dir is not None:
            for ext in self.distribution.ext_modules:
                if hasattr(ext, 'codegen_dir'):
                    src_dirs = []
                    build_dirs = []
                    filenames = []
                    for filepath in glob(
                        os.path.join(ext.codegen_dir, 'codegen_src/**'),
                        recursive=True,
                    ):
                        if os.path.isfile(filepath):
                            dirname = os.path.dirname(filepath)
                            dirpath = os.path.relpath(dirname, ext.codegen_dir)
                            src_dirs.append(os.path.join(ext.codegen_dir, dirpath))
                            build_dirs.append(os.path.join(codegen_build_dir, dirpath))
                            filenames.append(os.path.basename(filepath))

                    if filenames:
                        for src_dir, build_dir, filename in zip(src_dirs, build_dirs, filenames):
                            self.data_files.append(
                                (
                                    'osqp.codegen',
                                    src_dir,
                                    build_dir,
                                    [filename],
                                )
                            )

        super().run()


class CmdCMakeBuild(build_ext):
    def run(self):
        super().run()
        # For editable installs, after the extension(s) have been built, copy the 'codegen_src' folder
        # from the temporary build folder to the source folder
        if self.editable_mode:
            codegen_src_folder = os.path.join(self.build_temp, 'codegen_src')
            codegen_target_folder = os.path.join('src', 'osqp', 'codegen', 'codegen_src')
            if os.path.exists(codegen_src_folder):
                if os.path.exists(codegen_target_folder):
                    shutil.rmtree(codegen_target_folder)
                shutil.copytree(codegen_src_folder, codegen_target_folder)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        thisdir = os.path.dirname(os.path.abspath(__file__))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON=ON',
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            f'-DPYTHON_INCLUDE_DIRS={get_python_inc()}',
            '-DOSQP_BUILD_UNITTESTS=OFF',
            '-DOSQP_USE_LONG=OFF',  # https://github.com/numpy/numpy/issues/5906
            # https://github.com/ContinuumIO/anaconda-issues/issues/3823
            f'-DOSQP_CUSTOM_PRINTING={thisdir}/cmake/printing.h',
            f'-DOSQP_CUSTOM_MEMORY={thisdir}/cmake/memory.h',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if system() == 'Windows':
            cmake_args += ['-G', 'Visual Studio 17 2022']
            # Finding the CUDA Toolkit on Windows seems to work reliably only if BOTH
            # CMAKE_GENERATOR_TOOLSET (-T) and CUDA_TOOLKIT_ROOT_DIR are supplied to cmake
            if 'CUDA_TOOLKIT_ROOT_DIR' in os.environ:
                cuda_root = os.environ['CUDA_TOOLKIT_ROOT_DIR']
                cmake_args += ['-T', f'cuda={cuda_root}']
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

        # Save the build folder as a custom attribute in the extension object,
        #   as we'll need it to package the codegen files as package_data later.
        ext.codegen_dir = self.build_temp

        _ext_name = ext.name.split('.')[-1]
        cmake_args.extend([f'-DOSQP_EXT_MODULE_NAME={_ext_name}'])

        # What variables from the environment do we wish to pass on to cmake as variables?
        cmake_env_vars = (
            'CMAKE_CUDA_COMPILER',
            'CUDA_TOOLKIT_ROOT_DIR',
            'MKL_DIR',
            'MKL_ROOT',
        )
        for cmake_env_var in cmake_env_vars:
            cmake_var = os.environ.get(cmake_env_var)
            if cmake_var:
                cmake_args.extend([f'-D{cmake_env_var}={cmake_var}'])

        if ext.cmake_args is not None:
            cmake_args.extend(ext.cmake_args)

        check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        super().build_extension(ext)


extras_require = {'dev': ['pytest>=6', 'torch', 'numdifftools', 'pre-commit']}

algebra = os.environ.get('OSQP_ALGEBRA_BACKEND', 'builtin')
assert algebra in ('builtin', 'mkl', 'cuda'), f'Unknown algebra {algebra}'
if algebra == 'builtin':
    package_name = 'osqp'
    ext_modules = [CMakeExtension('osqp.ext_builtin', cmake_args=['-DOSQP_ALGEBRA_BACKEND=builtin'])]
    extras_require['mkl'] = ['osqp-mkl']
    extras_require['cuda'] = ['osqp-cuda']
else:
    package_name = f'osqp_{algebra}'
    ext_modules = [CMakeExtension(f'osqp_{algebra}', cmake_args=[f'-DOSQP_ALGEBRA_BACKEND={algebra}'])]


setup(
    name=package_name,
    author='Bartolomeo Stellato, Goran Banjac',
    author_email='bartolomeo.stellato@gmail.com',
    description='OSQP: The Operator Splitting QP Solver',
    long_description=open('README.rst').read(),
    package_dir={'': 'src'},
    # package_data for 'osqp.codegen' is populated by CustomBuildPy to include codegen_src files
    #   after building extensions, so it should not be included here.
    # It is however ok to specify package_data for submodules of 'osqp.codegen'.
    package_data={'osqp.codegen.pywrapper': ['*.jinja']},
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy>=1.7', 'scipy>=0.13.2', 'qdldl', 'jinja2'],
    python_requires='>=3.7',
    extras_require=extras_require,
    license='Apache 2.0',
    url='https://osqp.org/',
    cmdclass={'build_ext': CmdCMakeBuild, 'build_py': CustomBuildPy},
    packages=find_namespace_packages(where='src'),
    ext_modules=ext_modules,
)
