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
4. Install YALMIP and MOSEK. In **both** MATLAB files, adapt the `addpath(genpath(...))` statements to their actual installation locations. The committed paths reference `..\YALMIP-master` and Windows MOSEK 9.0/9.2 installations.
5. Create the output directories before running experiments:

   ```bash
   python -c "from pathlib import Path; [Path(p).mkdir(exist_ok=True) for p in ('Results', 'saved_weights')]"
   ```

6. For MNIST training, also adapt the hard-coded `c:/tmp/Lipparameters*.mat` paths and create their parent directory. The MNIST loaders use `/tmp` as their dataset root; adjust it if needed. Dataset downloads require network access.

## Start with the 2D comparison

From the repository root, after configuring the dependencies and MATLAB paths:

```bash
python Vergleich2D.py
```

This runs the existing comparison experiment, rather than a short smoke test. It trains multiple models, invokes MATLAB optimization, displays plots, and writes model files such as `2D_NomModel.pt`, `2D_L2Model.pt`, `2D_LipModel.pt`, and `2D_LMTModel.pt`. It also writes loss/Lipschitz plots and a results `.mat` file under `Results/`. Interactive plot windows may need to be closed for execution to continue.

The tuning scripts perform hyperparameter sweeps and can take substantially longer. Review their settings before starting them.

For the noise evaluation, first generate the four `MNIST196_*Model.pt` files with `VergleichMNIST196.py`, then run:

```bash
python "Test MNISTwithNoise.py"
```

## Data flow and current limitations

- Importing `solve_SDP_multi` starts a local MATLAB session.
- The bridge writes `weights.json` and `biases.json` in the working directory; MATLAB reads those files. Run from the repository root and **run only one experiment at a time in that directory** to avoid file collisions.
- Results and model filenames can be overwritten by subsequent runs. Copy outputs elsewhere before rerunning.
- Random seeds are not consistently fixed, so exact numerical reproduction is not guaranteed.
- Solver return statuses are not checked before results are consumed. Inspect solver output and validate numerical results.
- Model checkpoints are serialized with `torch.save(model, ...)`. Use only trusted checkpoints and a compatible environment.
- No end-to-end execution in a newly configured environment has been verified as part of this documentation update.

## Citation and reuse

If you use this code in research, cite the accompanying paper, **"Training robust neural networks using Lipschitz bounds"**. Full bibliographic metadata is not yet included here.

No license file is currently provided; redistribution and reuse permissions should be clarified with the authors.
