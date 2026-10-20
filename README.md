# cerebras-pytorch-demo

## Dependencies

- install [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Repo setup

```bash
# create an uv project with python 3.11
mkdir cerebras-pytorch-demo
cd cerebras-pytorch-demo
git init
uv init --bare --python '>=3.11,<3.12'
uv python pin 3.11
uv sync
uv run python --version # Python 3.11.x

# add Cerebras PyTorch (must match the cluster's software release, matches 3.2)
uv add cerebras_pytorch==2.10.0

# add Cerebras modelzoo as shallow submodule (only the release commit)
git clone --depth 1 --branch Release_2.10.0 https://github.com/Cerebras/modelzoo.git modelzoo
git submodule add https://github.com/Cerebras/modelzoo.git modelzoo
git config -f .gitmodules submodule.modelzoo.shallow true
git config -f .gitmodules submodule.modelzoo.ignore untracked
uv add --editable ./modelzoo --no-workspace

# test environment
uv run python -c "
import sys
import platform
from importlib.metadata import version

import torch
import cerebras.pytorch as cstorch
import cerebras.modelzoo as modelzoo

from cerebras.modelzoo import Trainer
from cerebras.modelzoo.trainer.callbacks import TrainingLoop

print('=== Environment ===')
print('Python:            ', sys.version.split()[0])
print('Platform:          ', platform.platform())
print('PyTorch:           ', torch.__version__)
print('Cerebras PyTorch:  ', version('cerebras-pytorch'))
print('Cerebras Model Zoo:', version('cerebras-modelzoo'))

print('\n=== Package Locations ===')
print('PyTorch:           ', torch.__file__)
print('Cerebras PyTorch:  ', cstorch.__file__)
print('Cerebras Model Zoo:', modelzoo.__file__)

print('\n=== API Checks ===')
print('cstorch.compile:   ', callable(cstorch.compile))
print('cstorch.trace:     ', hasattr(cstorch, 'trace'))
print('cstorch.optim.SGD: ', callable(cstorch.optim.SGD))
print('Trainer:           ', callable(Trainer))
print('TrainingLoop:      ', callable(TrainingLoop))

print('\n=== Hardware ===')
print('CUDA available:    ', torch.cuda.is_available())
print('CUDA devices:      ', torch.cuda.device_count())

print('\nAll imports successful!')
"
```

## Run training

```bash
# on CPU
uv run python train.py --device cpu

# on GPU
uv run python train.py --device gpu

# on Cerebras CS-3
uv run python train.py --device csx
```

## View training logs

```bash
uv add --dev tensorboard "setuptools<82"
uv run tensorboard --logdir runs/cpu # for CPU training logs
```
