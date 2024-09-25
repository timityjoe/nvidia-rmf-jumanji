
# See
# https://github.com/instadeepai/jumanji
# https://instadeepai.github.io/jumanji/guides/training/

# Conda Setup
conda init bash
conda create --name conda310-jumanji python=3.10
conda activate conda310-jumanji
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Setup
export PYTHONPATH="${PYTHONPATH}:/media/timityjoe/Data2/Cube/nvidia-rmf-jumanji/"
source jumanup.sh

## Pull additional repos
vcs import < og_marl_repo.txt
pip install https://github.com/instadeepai/Mava/archive/refs/tags/0.1.2.zip
pip3 install numpy --upgrade
pip install dm-reverb[tensorflow]
pip install protobuf==3.20.0


# Pip setup (from pyproject.toml and setup.py)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113^C
python -m pip install .
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
pip install -r requirements/requirements-train.txt
pip install -r requirements/requirements_og_marl.txt

pip install jax==0.4.23 jaxlib==0.4.23
pip install scipy==1.7.2
pip install numpy==1.23

pip install id-mava[reverb,tf]==0.1.3
pip install dm-acme
pip install dm-reverb
pip install cpprb
pip install flax
pip install flashbax
pip install .

Relevant directories:
{$JUMANJI}/jumanji/training
{$JUMANJI}/jumanji/environments/routing/robot_warehouse
{$JUMANJI}/jumanji/training/networks/robot_warehouse
{$JUMANJI}/jumanji/training/configs/env/robot_warehouse.yaml
{$JUMANJI}/jumanji/training/train.py

python3 -m jumanji.training.train
python3 -m jumanji.training.train_rware
python3 -m jumanji.training.train_rware_a2c
python3 -m jumanji.training.train_multicvrp
python3  jumanji/training/train_multicvrp.py
python3 -m jumanji.training.test
python3  jumanji/training/test.py
python3 -m jumanji.training.load_checkpoint


# Offline data generation & training (from og_marl_old)
python3 -m jumanji.training.1_rware_a2c_gen_dataset 
python3 -m jumanji.training.1_rware_a2c_gen_dataset --algo_name=qmix --dataset_quality=Good --env_name=rware
python3 -m jumanji.training.1_rware_a2c_gen_dataset --algo_name=maicq --dataset_quality=Good --env_name=rware

python3 -m jumanji.training.2_rware_a2c_pretrain_offline_network
python3 -m jumanji.training.3_rware_a2c_online_finetune


# Start Tensorboard
cd ./a2c_robot_warehouse
tensorboard --logdir=./ --port=8080


