# @title Set up JAX for available hardware (run me) { display-mode: "form" }

import subprocess
import os
from tqdm import tqdm

# Based on https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system
try:
    subprocess.check_output('nvidia-smi')
    print("a GPU is connected.")
except Exception:
    # TPU or CPU
    if "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]:
        import jax.tools.colab_tpu

        jax.tools.colab_tpu.setup_tpu()
        print("A TPU is connected.")
    else:
        print("Only CPU accelerator is connected.")

import pickle

import jax
from huggingface_hub import hf_hub_download
from hydra import compose, initialize

from jumanji.training.setup_train import setup_agent, setup_env
from jumanji.training.utils import first_from_device

# %matplotlib notebook

# env = "multi_cvrp"  # @param ['bin_pack', 'cleaner', 'connector', 'cvrp', 'game_2048', 'graph_coloring', 'job_shop', 'knapsack', 'maze', 'minesweeper', 'mmst', 'multi_cvrp', 'robot_warehouse', 'rubiks_cube', 'snake', 'sudoku', 'tetris', 'tsp']
env = "robot_warehouse"
# env = "maze"
# env = "cleaner"
# env = "bin_pack"
# env = "snake"

# agent = "random"  # @param ['random', 'a2c']
agent = "a2c"

config_url = "configs/env/{env}.yaml"
env_url = "configs/env/{env}.yaml"

#@title Download Jumanji Configs (run me) { display-mode: "form" }

import os
import requests

# def download_file(url: str, file_path: str) -> None:
#     # Send an HTTP GET request to the URL
#     response = requests.get(url)
#     # Check if the request was successful (status code 200)
#     if response.status_code == 200:
#         with open(file_path, "wb") as f:
#             f.write(response.content)
#     else:
#         print("Failed to download the file.")

# os.makedirs("configs", exist_ok=True)
# config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
# download_file(config_url, "configs/config.yaml")
# env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env}.yaml"
# os.makedirs("configs/env", exist_ok=True)
# download_file(env_url, f"configs/env/{env}.yaml")

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="config.yaml", overrides=[f"env={env}", f"agent={agent}"])
cfg

# Chose the corresponding checkpoint from the InstaDeep Model Hub
# https://huggingface.co/InstaDeepAI
REPO_ID = f"InstaDeepAI/jumanji-benchmark-a2c-{cfg.env.registered_version}"
FILENAME = f"{cfg.env.registered_version}_training_state"

# model_checkpoint = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
# model_checkpoint = "{$JUMANJI}/training_state"
model_checkpoint = "training_state"
# print(f"model_checkpoint:{model_checkpoint}")
path = os.getcwd()
print("PWD:", path)

with open(model_checkpoint, "rb") as f:
    training_state = pickle.load(f)
print(f"training_state:{training_state}")
# print(f"training_state.params_state:{training_state.params_state}")
# print(f"training_state.params_state.params:{training_state.params_state.params}")

# Mod by Tim: Previous was from a2c_agent.py; this looks to be random_agent.py
params = first_from_device(training_state.params_state.params)
# params = first_from_device(training_state.acting_state.state)

env = setup_env(cfg).unwrapped
agent = setup_agent(cfg, env)
policy = jax.jit(agent.make_policy(params.actor, stochastic = False))
if agent == "a2c":
    policy = lambda *args: policy(*args)[0]


NUM_EPISODES = 10

reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)
states = []
key = jax.random.PRNGKey(cfg.seed)
for episode in tqdm(range(NUM_EPISODES)):
    key, reset_key = jax.random.split(key) 
    state, timestep = reset_fn(reset_key)
    states.append(state)
    while not timestep.last():
        key, action_key = jax.random.split(key)
        observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
        action = policy(observation, action_key)
        state, timestep = step_fn(state, action.squeeze(axis=0))
        env.render(state)
        states.append(state)
    # Freeze the terminal frame to pause the GIF.
    for _ in range(3):
        states.append(state)

print(f"Animating, len states:{len(states)}")     
env.animate(states, interval=150)
print("End!")    