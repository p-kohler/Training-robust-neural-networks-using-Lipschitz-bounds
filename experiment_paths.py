"""Portable input paths and one unique output directory per Python process."""

import os
import tempfile
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parent
_run_dir = None


def configured_path(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser()
    return (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def data_dir():
    path = configured_path('LIP_DATA_DIR', REPO_ROOT / 'data')
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_dir():
    global _run_dir
    if _run_dir is None:
        root = configured_path('LIP_OUTPUT_ROOT', REPO_ROOT / 'runs')
        root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime('%Y%m%d-%H%M%S-')
        _run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))
        print('Experiment output: {}'.format(_run_dir), flush=True)
    return _run_dir


def output_path(relative):
    """Return an output filename inside this run, creating parent directories."""
    path = (run_dir() / relative).resolve()
    if run_dir() not in path.parents:
        raise ValueError('Output paths must stay inside the run directory.')
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def model_path(filename):
    """Read explicit training-run checkpoints; retain legacy root fallback."""
    return configured_path('LIP_MODEL_DIR', REPO_ROOT) / filename
