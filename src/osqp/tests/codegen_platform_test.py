import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from typing import Any

from setuptools import Distribution


class codegen_platform_tests(unittest.TestCase):
    def configure(self, platform, generator=None):
        template = Path(__file__).parents[1] / 'codegen' / 'pywrapper' / 'setup.py.jinja'
        namespace: dict[str, Any] = {'__file__': str(template)}
        source = template.read_text().replace('{{extension_name}}', 'test_ext').replace('{{embedded_mode}}', '1')
        with patch('setuptools.setup'):
            exec(compile(source, str(template), 'exec'), namespace)
        command = namespace['CmdCMakeBuild'](Distribution())
        command.plat_name = platform
        command.debug = False
        extension = namespace['CMakeExtension']('test_ext')
        env = dict(os.environ)
        env.pop('CMAKE_GENERATOR', None)
        if generator is not None:
            env['CMAKE_GENERATOR'] = generator
        with tempfile.TemporaryDirectory() as directory:
            command.build_temp = str(Path(directory) / 'build')
            command.build_lib = str(Path(directory) / 'lib')
            with patch.dict(os.environ, env, clear=True), patch.dict(namespace, {'system': lambda: 'Windows'}):
                with patch.dict(namespace, {'check_call': Mock()}) as mocked:
                    command.build_extension(extension)
                    return [call.args[0] for call in mocked['check_call'].call_args_list]

    def test_native_arm64_visual_studio(self):
        configure, build = self.configure('win-arm64')
        self.assertEqual(configure[configure.index('-A') + 1], 'ARM64')
        self.assertNotIn('/m', build)

    def test_emulated_amd64_interpreter(self):
        configure, _ = self.configure('win-amd64')
        self.assertEqual(configure[configure.index('-A') + 1], 'x64')

    def test_win32_interpreter(self):
        configure, _ = self.configure('win32')
        self.assertEqual(configure[configure.index('-A') + 1], 'Win32')

    def test_explicit_visual_studio_generator(self):
        configure, _ = self.configure('win-arm64', 'Visual Studio 18 2026')
        self.assertEqual(configure[configure.index('-G') + 1], 'Visual Studio 18 2026')
        self.assertEqual(configure[configure.index('-A') + 1], 'ARM64')

    def test_ninja_does_not_receive_visual_studio_options(self):
        for generator in ('Ninja', 'Ninja Multi-Config'):
            with self.subTest(generator=generator):
                configure, build = self.configure('win-arm64', generator)
                self.assertEqual(configure[configure.index('-G') + 1], generator)
                self.assertNotIn('-A', configure)
                self.assertNotIn('/m', build)
                self.assertIn('--parallel', build)


if __name__ == '__main__':
    unittest.main()
