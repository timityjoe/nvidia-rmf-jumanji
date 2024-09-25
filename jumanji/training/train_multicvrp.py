# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import subprocess
import os
import requests



# @title Set up JAX for available hardware (run me) { display-mode: "form" }
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


import warnings
warnings.filterwarnings("ignore")

from jumanji.training.train import train
from hydra import compose, initialize


# env = "maze"  # @param ['bin_pack', 'cleaner', 'connector', 'cvrp', 'game_2048', 'graph_coloring', 'job_shop', 'knapsack', 'maze', 'minesweeper', 'mmst', 'multi_cvrp', 'robot_warehouse', 'rubiks_cube', 'snake', 'sudoku', 'tetris', 'tsp']
env = "multi_cvrp"
# agent = "random"  # @param ['random', 'a2c']
agent = "a2c"

print(f"env:{env}, agent:{agent}")


#@title Download Jumanji Configs (run me) { display-mode: "form" }

def download_file(url: str, file_path: str) -> None:
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print("Failed to download the file.")

if __name__ == "__main__":
    print("0) Start")
    os.makedirs("configs", exist_ok=True)
    config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
    download_file(config_url, "configs/config.yaml")
    env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env}.yaml"
    os.makedirs("configs/env", exist_ok=True)
    download_file(env_url, f"configs/env/{env}.yaml")

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config.yaml", overrides=[f"env={env}", f"agent={agent}", "logger.type=terminal", "logger.save_checkpoint=true"])

    print(f"1) cfg:{cfg}")
    train(cfg)



# def main(args = None):
#     print("0) Start")

#     os.makedirs("configs", exist_ok=True)
#     config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
#     download_file(config_url, "configs/config.yaml")
#     env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env}.yaml"
#     os.makedirs("configs/env", exist_ok=True)
#     download_file(env_url, f"configs/env/{env}.yaml")

#     with initialize(version_base=None, config_path="configs"):
#         cfg = compose(config_name="config.yaml", overrides=[f"env={env}", f"agent={agent}", "logger.type=terminal", "logger.save_checkpoint=true"])

#     print(f"1) cfg:{cfg}")

#     # train(cfg)

# main()


