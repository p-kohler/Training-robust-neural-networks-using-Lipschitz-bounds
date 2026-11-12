# Training robust neural networks using Lipschitz bounds

Research code accompanying the paper **"Training robust neural networks using Lipschitz bounds"**.

The experiments compare nominal training, L2 regularization, Lipschitz-based training, and Lipschitz-margin training (LMT). PyTorch performs neural-network training; MATLAB, YALMIP, and an SDP solver are used for Lipschitz-bound calculations and optimization.

## Repository guide

| File | Purpose |
| --- | --- |
| `Vergleich2D.py` | Compare training methods on synthetic 2D data |
| `VergleichMNIST196.py`, `VergleichMNIST784.py` | Compare methods for the 196- and 784-input MNIST experiments |
| `Tuning2D.py`, `TuningMNIST196.py`, `TuningMNIST784.py` | Sweep Lipschitz-training hyperparameters |
| `Test MNISTwithNoise.py` | Evaluate saved MNIST196 models with perturbed inputs |
| `solve_SDP_multi.py` | Python-to-MATLAB bridge and optimization-parameter conversion |
| `calculate_Lipschitz_multi.m` | Calculate a Lipschitz bound using a semidefinite program |
| `solve_sdp_multi.m` | Update network parameters through semidefinite optimization |

## Requirements

- Python with `torch`, `torchvision`, `torchsummary`, `numpy`, `scipy`, and `matplotlib`.
- MATLAB with MATLAB Engine for Python installed for the chosen MATLAB/Python combination.
- [YALMIP](https://yalmip.github.io/download/) and [MOSEK](https://www.mosek.com/downloads/).
- Working MATLAB and MOSEK licenses, configured outside this repository.

The original tested dependency versions are not recorded. This is legacy research code, not a verified installation recipe for current package releases. In particular, the scripts use `np.object`, which is unavailable in recent NumPy releases. Use a compatible legacy environment or migrate those calls before using a modern NumPy environment. Match PyTorch and torchvision versions and check MATLAB Engine compatibility with your Python version.

## Setup

1. Clone the repository and open a terminal in its root directory.
2. Create a Python environment and install the dependencies listed above, choosing mutually compatible versions.
3. Install MATLAB Engine for Python according to the instructions shipped with your MATLAB release.
4. Install YALMIP and MOSEK. Either make them available through MATLAB's startup path, or set `YALMIP_PATH` and `MOSEK_PATH` before starting Python. These variables must point to existing toolbox directories (use absolute paths); the MATLAB helpers add them recursively. No source-file edits are needed. Solver selection remains YALMIP's default.
5. Output and dataset directories are created automatically. MNIST downloads require network access.

### Path configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LIP_OUTPUT_ROOT` | `runs/` next to the scripts | Parent of unique run directories |
| `LIP_DATA_DIR` | `data/` next to the scripts | Shared MNIST dataset cache |
| `LIP_MODEL_DIR` | Repository root (legacy checkpoints) | Input checkpoints for noise evaluation |
| `YALMIP_PATH` | Existing MATLAB path | YALMIP installation directory |
| `MOSEK_PATH` | Existing MATLAB path | MOSEK MATLAB/toolbox installation directory |

Relative `LIP_*` paths are resolved against the repository, not the terminal's working directory. `~` is expanded. Set variables in the shell before launching Python, for example on Linux/macOS:

```bash
export YALMIP_PATH="/absolute/path/to/YALMIP"
export MOSEK_PATH="/absolute/path/to/mosek/toolbox"
python Vergleich2D.py
```

In PowerShell use `$env:YALMIP_PATH = 'C:\path\to\YALMIP'` and the equivalent for MOSEK. Toolboxes already configured in MATLAB do not require these variables.

## Start with the 2D comparison

From the repository root, after configuring the dependencies and MATLAB paths:

```bash
python Vergleich2D.py
```

This runs the existing comparison experiment, rather than a short smoke test. It trains multiple models, invokes MATLAB optimization, displays plots, and writes model files such as `2D_NomModel.pt`, `2D_L2Model.pt`, `2D_LipModel.pt`, and `2D_LMTModel.pt`. All outputs are placed in the unique directory printed as `Experiment output: ...`: model checkpoints at its top level, plots/results under `Results/`, and MATLAB exchange files alongside them. The root `runs/` and `data/` directories are ignored by Git. Interactive plot windows may need to be closed for execution to continue.

The tuning scripts perform hyperparameter sweeps and can take substantially longer. Review their settings before starting them.

For the noise evaluation, first generate the four trusted checkpoints `MNIST196_NomModel.pt`, `MNIST196_L2Model.pt`, `MNIST196_LipModel.pt`, and `MNIST196_LMTModel.pt` with `VergleichMNIST196.py`. Set `LIP_MODEL_DIR` to the printed training-run directory, then run (Linux/macOS example):

```bash
LIP_MODEL_DIR="runs/<training-run-directory>" python "Test MNISTwithNoise.py"
```

The evaluation itself needs only PyTorch and torchvision; it does not start MATLAB. It loads the full-model checkpoints on the CPU and downloads the MNIST test set to `LIP_DATA_DIR` if needed. Only load trusted checkpoints, since full-model loading can execute code. It prints test accuracy for each model over Gaussian standard deviations from 0 to 0.5 and uniform full widths from 0 to 2 (noise in `[-b/2, b/2)`). All four models receive the same noisy images at each setting, with a fixed seed. Noise is added without clipping pixels to `[0, 1]`. No plots or result files are written by this evaluation.

## Data flow and current limitations

- MATLAB starts lazily on the first solver call and is closed at process exit. The bridge adds the repository to MATLAB's path and sets MATLAB's working directory to the run folder; Python's working directory is unchanged.
- Each Python process creates one unique run directory on first output, even when processes start concurrently. Run separate experiments in separate processes. Concurrent threads within one process are not supported.
- `weights.json`, `biases.json`, and intermediate `Lipparameters*.mat` files stay inside that run. Within a run, intermediate files are intentionally overwritten; the legacy `saved_weights/mnist_weights.mat` also retains only its most recent write. Model and result filenames otherwise retain their original naming.
- Noise evaluation reads explicitly selected checkpoints and does not create a training run. For old checkpoints kept in the repository root, omit `LIP_MODEL_DIR`.
- Random seeds are not consistently fixed, so exact numerical reproduction is not guaranteed.
- Both MATLAB optimization routines check YALMIP diagnostics before consuming solver values. Any nonzero status raises `Lipschitz:SolverFailure` with the operation, code, and diagnostic message. A successful status still requires scientific validation of the result.
- Model checkpoints are serialized with `torch.save(model, ...)`. Use only trusted checkpoints and a compatible environment.
- No end-to-end execution in a newly configured environment has been verified as part of this update. Python path/isolation and mocked MATLAB-bridge tests do not validate numerical solver behavior.

## Checks without MATLAB

```bash
python -m unittest discover -s tests -v
```

These checks cover portable paths, separate-process output isolation, script output routing, and MATLAB bridge setup/cleanup with a fake engine. They do not run training or an SDP solver.

## Citation and reuse

If you use this code in research, cite the accompanying paper, **"Training robust neural networks using Lipschitz bounds"**. Full bibliographic metadata is not yet included here.

No license file is currently provided; redistribution and reuse permissions should be clarified with the authors.

