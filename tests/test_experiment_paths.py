"""Exercise filesystem behavior and bridge boundaries without MATLAB or training."""
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_paths


class PathsTest(unittest.TestCase):
    def test_separate_processes_from_another_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='lip paths ') as root:
            env = dict(os.environ, PYTHONPATH=str(ROOT), LIP_OUTPUT_ROOT=root)
            command = [sys.executable, '-c',
                       'from experiment_paths import output_path; '
                       'p=output_path("nested/result.txt"); '
                       'open(p,"w").write("result"); print(p)']
            processes = [subprocess.Popen(command, cwd=root, env=env,
                                          stdout=subprocess.PIPE, text=True)
                         for _ in range(2)]
            paths = []
            for process in processes:
                output, _ = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0)
                paths.append(Path(output.splitlines()[-1]))
            self.assertNotEqual(paths[0], paths[1])
            for path in paths:
                self.assertEqual(path.read_text(), 'result')
                self.assertEqual(path.parents[2], Path(root))

    def test_relative_configuration_and_model_input(self):
        with patch.dict(os.environ, {'LIP_MODEL_DIR': 'runs/example',
                                     'LIP_DATA_DIR': 'data/example'}):
            self.assertEqual(experiment_paths.model_path('a.pt'),
                             ROOT / 'runs/example/a.pt')
            self.assertEqual(experiment_paths.configured_path('LIP_DATA_DIR', ''),
                             ROOT / 'data/example')

    def test_output_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(experiment_paths, '_run_dir', None), \
                 patch.dict(os.environ, {'LIP_OUTPUT_ROOT': root}):
                with self.assertRaises(ValueError):
                    experiment_paths.output_path('../escape.txt')

    def test_training_scripts_route_file_outputs(self):
        for path in list(ROOT.glob('Tuning*.py')) + list(ROOT.glob('Vergleich*.py')):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if (isinstance(node.func, ast.Attribute)
                        and node.func.attr in ('save', 'savefig')):
                    index = 1 if node.func.attr == 'save' else 0
                    self.assertIsInstance(node.args[index], ast.Call, str(path))
                    self.assertEqual(node.args[index].func.id, 'output_path')
            self.assertNotIn('c:/tmp', path.read_text())
            self.assertNotIn("MNIST('/tmp'", path.read_text())


class BridgeTest(unittest.TestCase):
    def test_lazy_engine_exchange_and_failure_cleanup(self):
        class Engine:
            def __init__(self, fail=False):
                self.calls = []
                self.fail = fail
            def addpath(self, path, **kwargs):
                self.calls.append(('addpath', path))
            def cd(self, path, **kwargs):
                self.calls.append(('cd', path))
            def configure_solver_paths(self, **kwargs):
                if self.fail:
                    raise RuntimeError('missing toolbox')
            def quit(self):
                self.calls.append(('quit',))
            def calculate_Lipschitz_multi(self, parameters, dims):
                self.exchange = json.loads(Path(self.calls[1][1], 'weights.json').read_text())
                return {'Lipschitz': 1.0}

        engine = Engine()
        matlab = types.ModuleType('matlab')
        matlab.engine = types.ModuleType('matlab.engine')
        matlab.engine.start_matlab = lambda: engine
        matlab.int64 = lambda x: x
        matlab.double = lambda x: x
        fake_modules = {'matlab': matlab, 'matlab.engine': matlab.engine,
                        'torch': types.ModuleType('torch')}
        with tempfile.TemporaryDirectory() as root, \
             patch.dict(sys.modules, fake_modules), \
             patch.dict(os.environ, {'LIP_OUTPUT_ROOT': root}), \
             patch.object(experiment_paths, '_run_dir', None), \
             patch('atexit.register') as register:
            sys.modules.pop('solve_SDP_multi', None)
            bridge = importlib.import_module('solve_SDP_multi')
            self.assertEqual(engine.calls, [])
            cwd = Path.cwd()
            self.assertEqual(bridge.build_T_multi([], [], [2, 1]), {'Lipschitz': 1.0})
            self.assertEqual(engine.exchange, {})
            self.assertEqual(engine.calls[0], ('addpath', str(ROOT)))
            self.assertEqual(Path(engine.calls[1][1]), experiment_paths.run_dir())
            self.assertEqual(Path.cwd(), cwd)
            self.assertIs(bridge.get_engine(), engine)
            register.assert_called_once()
            bad_engine = Engine(fail=True)
            matlab.engine.start_matlab = lambda: bad_engine
            bridge._engine = None
            with self.assertRaisesRegex(RuntimeError, 'missing toolbox'):
                bridge.get_engine()
            self.assertIn(('quit',), bad_engine.calls)
            self.assertIsNone(bridge._engine)
            sys.modules.pop('solve_SDP_multi', None)


if __name__ == '__main__':
    unittest.main()
