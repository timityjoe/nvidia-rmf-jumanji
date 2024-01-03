
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
cd ./a2c_robot_warehouse
tensorboard --logdir=./ --port=8080


