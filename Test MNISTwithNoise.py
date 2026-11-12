"""Evaluate trusted MNIST196 full-model checkpoints on noisy MNIST images."""

import inspect
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms


BATCH_SIZE = 100
STD_VALUES = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
WIDTH_VALUES = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2]


class Network(nn.Module):
    """Inference methods for Network objects saved by VergleichMNIST196.py.

    Full-model checkpoints restore AvgPool and net without calling __init__.
    Keep this class in __main__ when running the script for pickle compatibility.
    """

    def forward(self, x):
        images = x.reshape(-1, 1, 28, 28)
        pooled = self.AvgPool(images)
        return self.net(pooled.reshape(images.size(0), 196))


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class AddUniformNoise:
    def __init__(self, a=0., b=1.):
        self.a = a
        self.b = b

    def __call__(self, tensor):
        # b is the full width: noise is uniform on [a-b/2, a+b/2).
        return tensor + (torch.rand_like(tensor) - 0.5) * self.b + self.a


def create_test_loader(std=None, b=None):
    if std is not None and b is not None:
        raise ValueError("Choose Gaussian std or uniform width b, not both.")
    steps = [transforms.ToTensor()]
    if std is not None:
        steps.append(AddGaussianNoise(std=std))
    elif b is not None:
        steps.append(AddUniformNoise(b=b))
    test_set = datasets.MNIST(
        '/tmp', train=False, download=True,
        transform=transforms.Compose(steps),
    )
    return torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
    )


def load_models():
    paths = {
        name: Path('MNIST196_{}Model.pt'.format(suffix))
        for name, suffix in [('Nominal', 'Nom'), ('L2', 'L2'),
                             ('Lipschitz', 'Lip'), ('LMT', 'LMT')]
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    # Full-model loading requires trusted files. Older PyTorch versions do not
    # expose weights_only; recent versions default to weights-only loading.
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    models = {}
    for name, path in paths.items():
        models[name] = torch.load(str(path), **options)
        models[name].eval()
    return models


def evaluate(models, test_loader):
    """Evaluate every model on the same realized noise for each image."""
    correct = {name: 0 for name in models}
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            total += labels.size(0)
            for name, model in models.items():
                predictions = model(data).argmax(dim=1)
                correct[name] += (predictions == labels).sum().item()
    if total == 0:
        raise ValueError("The test dataset is empty.")
    return {name: 100. * count / total for name, count in correct.items()}


def main():
    models = load_models()
    for kind, values in [('std', STD_VALUES), ('b', WIDTH_VALUES)]:
        for value in values:
            # Repeatable noise, shared by all models within this evaluation.
            torch.manual_seed(0)
            loader = create_test_loader(**{kind: value})
            print('{} = {}'.format(kind, value))
            for name, accuracy in evaluate(models, loader).items():
                print('{} NN: Test Accuracy: {:.3f} %'.format(name, accuracy))


if __name__ == '__main__':
    main()
