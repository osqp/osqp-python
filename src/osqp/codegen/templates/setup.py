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


class CmdCMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if system() == "Windows":
            cmake_args += ['-G', 'Visual Studio 17 2022']
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if os.path.exists(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp)

        if ext.cmake_args is not None:
            cmake_args.extend(ext.cmake_args)

        check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        super().build_extension(ext)


setup(
    name='foo',
    author='Bartolomeo Stellato, Goran Banjac',
    author_email='bartolomeo.stellato@gmail.com',
    description='OSQP: The Operator Splitting QP Solver',
    license='Apache 2.0',
    url="https://osqp.org/",

    python_requires='>=3.7',
    setup_requires=["numpy >= 1.7"],
    install_requires=['numpy >= 1.7'],

    ext_modules=[CMakeExtension('foo', cmake_args=['-DEMBEDDED=1'])],
    cmdclass={'build_ext': CmdCMakeBuild},
    zip_safe=False
)