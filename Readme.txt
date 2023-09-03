
# See
# https://github.com/instadeepai/jumanji
# https://instadeepai.github.io/jumanji/guides/training/


# Conda Setup
conda init bash
conda create --name conda39-jumanji python=3.9
conda activate conda39-jumanji
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


# Pip setup (from pyproject.toml and setup.py)
python -m pip install .
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
pip install -r requirements/requirements-train.txt


# Python3.7 setup
$DATA/anaconda3/envs/isaac-sim-2022-2-1/lib/python3.7/site-packages/haiku/_src/layer_stack.py, L20
$NV_ISAAC_SIM/isaac_sim-2022.2.0/exts/omni.rmf.demos/omni/rmf/nvidia-rmf-jumanji/jumanji/training/timer.py
	- "from typing" to "from typing_extensions"
pip3 install typing-extensions --upgrade

Relevant directories:
{$JUMANJI}/jumanji/training
{$JUMANJI}/jumanji/environments/routing/robot_warehouse
{$JUMANJI}/jumanji/training/networks/robot_warehouse
{$JUMANJI}/jumanji/training/configs/env/robot_warehouse.yaml
{$JUMANJI}/jumanji/training/train.py

python3 -m jumanji.training.train
python3 -m jumanji.training.train_rware
python3 -m jumanji.training.train_rware_a2c
python3 -m jumanji.training.load_checkpoint



# Start Tensorboard
tensorboard --logdir=./ --port=8080


