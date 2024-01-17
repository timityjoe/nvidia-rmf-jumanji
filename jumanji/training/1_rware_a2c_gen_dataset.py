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

import functools
import logging
from typing import Dict, Tuple

import hydra
import jax
import jax.numpy as jnp
import omegaconf
from tqdm.auto import trange

from jumanji.training import utils

from jumanji.training.agents.random import RandomAgent
# from jumanji.training.agents.a2c import A2CAgent

from jumanji.training.loggers import TerminalLogger
from jumanji.training.setup_train import (
    setup_agent,
    setup_env,
    setup_evaluators,
    setup_logger,
    setup_training_state,
)
from jumanji.training.timer import Timer
from jumanji.training.types import TrainingState

import os
import requests

import warnings
warnings.filterwarnings("ignore")

from jumanji.training.train import train
from hydra import compose, initialize

# @title Set up JAX for available hardware (run me) { display-mode: "form" }
import subprocess
import os

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Mod by Tim:
from loguru import logger as loguru_logger
# loguru_logger.remove()
# loguru_logger.add(sys.stdout, level="INFO")
# loguru_logger.add(sys.stdout, level="SUCCESS")
# loguru_logger.add(sys.stdout, level="WARNING")

# This has been changed, to support tf2 and jax in the new og_marl. 
# See       /og_marl/tf2/systems/base.py
# Also      /og_marl/examples/tf2/online/qmix_pursuit.py

from og_marl_tjt.og_marl.loggers import JsonWriter

# TF2 based systems
# from og_marl_tjt.og_marl.tf2.systems.qmix import QMIXSystem

# Switch to JAX - See https://github.com/instadeepai/og-marl
from og_marl_tjt.og_marl.environments.rware import Warehouse, RewardType
from og_marl_tjt.og_marl.jax.systems.maicq2 import train_maicq_system

from og_marl_tjt.og_marl.replay_buffers import SequenceCPPRB
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "voltage_control", "Environment name.")
flags.DEFINE_string("scenario", "case33_3min_final", "Environment scenario name.")
flags.DEFINE_string("dataset", "Replay", "Dataset type.: 'Good', 'Medium', 'Poor' or '' for combined. ")
flags.DEFINE_string("system", "iddpg", "System name.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_float("trainer_steps", 1e5, "Number of training steps.")
flags.DEFINE_integer("batch_size", 32, "Number of training steps.")
flags.DEFINE_integer("num_offline_sequences", 10_000, "Number of sequences to load from the offline dataset into the replay buffer.")
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


def download_file(url: str, file_path: str) -> None:
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print("Failed to download the file.")

# env = "multi_cvrp"  # @param ['bin_pack', 'cleaner', 'connector', 'cvrp', 'game_2048', 'graph_coloring', 'job_shop', 'knapsack', 'maze', 'minesweeper', 'mmst', 'multi_cvrp', 'robot_warehouse', 'rubiks_cube', 'snake', 'sudoku', 'tetris', 'tsp']
env = "robot_warehouse"
# env = "maze"
# env = "cleaner"
# env = "snake"

# agent = "random"  # @param ['random', 'a2c']
agent = "a2c"

config_url = "configs/env/{env}.yaml"
env_url = "configs/env/{env}.yaml"


# @hydra.main(config_path="configs", config_name="config.yaml")
@hydra.main(config_path="configs/env", config_name="robot_warehouse.yaml")
def train(cfg: omegaconf.DictConfig, log_compiles: bool = False) -> None:
    print(f"train() yaml:{omegaconf.OmegaConf.to_yaml(cfg)}")
    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    logging.info({"devices": jax.local_devices()})

    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    logger = setup_logger(cfg)
    env = setup_env(cfg)

    #---------------------------------------------------------------------------------
    # Tensorflow2 based system
    # Convert the environment for datalogging; for offline pre-training purposes
    # See https://arxiv.org/pdf/2302.00521.pdf
    # Also      /og_marl/examples/tf2/online/qmix_pursuit.py
    # loguru_logger.info("1) Setup OG-MARL Offline Data Logger")

    # json_writer = JsonWriter('logs/rware', 'tf2system', 'Good', 'rware', '')
    # system = QMIXSystem(env, json_writer, add_agent_id_to_obs=True)
    # replay_buffer = SequenceCPPRB(env, batch_size=64)
    # system.train_online(replay_buffer)
    #---------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------
    dataset_path = "datasets/rware/Good"

    # Instantiate environment for evaluation
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)

    # Setup a logger to write to terminal
    logger = TerminalLogger()

    # Train system
    train_maicq_system(env, logger, dataset_path)
    #---------------------------------------------------------------------------------










    # agent = setup_agent(cfg, env)
    # stochastic_eval, greedy_eval = setup_evaluators(cfg, agent)
    # training_state = setup_training_state(env, agent, init_key)
    # num_steps_per_epoch = (
    #     cfg.env.training.n_steps
    #     * cfg.env.training.total_batch_size
    #     * cfg.env.training.num_learner_steps_per_epoch
    # )

    # loguru_logger.info(f"num_steps_per_epoch:{num_steps_per_epoch}")

    # eval_timer = Timer(out_var_name="metrics")
    # train_timer = Timer(
    #     out_var_name="metrics", num_steps_per_timing=num_steps_per_epoch
    # )

    # @functools.partial(jax.pmap, axis_name="devices")
    # def epoch_fn(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
    #     loguru_logger.info("epoch_fn(), reading in [training_state] object...")
    #     training_state, metrics = jax.lax.scan(
    #         lambda training_state, _: agent.run_epoch(training_state),
    #         training_state,
    #         None,
    #         cfg.env.training.num_learner_steps_per_epoch,
    #     )
    #     metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    #     return training_state, metrics

    # with jax.log_compiles(log_compiles), logger:
    #     for i in trange(
    #         cfg.env.training.num_epochs,
    #         disable=isinstance(logger, TerminalLogger),
    #     ):
    #         env_steps = i * num_steps_per_epoch
    #         loguru_logger.info(f"i:{i}, num_epochs:{cfg.env.training.num_epochs}, env_steps:{env_steps}")

    #         # Evaluation
    #         key, stochastic_eval_key, greedy_eval_key = jax.random.split(key, 3)
    #         # Stochastic evaluation
    #         with eval_timer:
    #             metrics = stochastic_eval.run_evaluation(
    #                 training_state.params_state, stochastic_eval_key
    #             )
    #             jax.block_until_ready(metrics)
    #         logger.write(
    #             data=utils.first_from_device(metrics),
    #             label="eval_stochastic",
    #             env_steps=env_steps,
    #         )
    #         if not isinstance(agent, RandomAgent):
    #             loguru_logger.info("A2C_Agent:")
    #             # Greedy evaluation
    #             with eval_timer:
    #                 metrics = greedy_eval.run_evaluation(
    #                     training_state.params_state, greedy_eval_key
    #                 )
    #                 jax.block_until_ready(metrics)
    #             logger.write(
    #                 data=utils.first_from_device(metrics),
    #                 label="eval_greedy",
    #                 env_steps=env_steps,
    #             )
    #             loguru_logger.info("End Stochastic evaluation - logger.write()")

    #         # Training
    #         with train_timer:
    #             loguru_logger.info(f"train_timer:{train_timer}")
    #             training_state, metrics = epoch_fn(training_state)
    #             jax.block_until_ready((training_state, metrics))
    #         logger.write(
    #             data=utils.first_from_device(metrics),
    #             label="train",
    #             env_steps=env_steps,
    #         )
    #         loguru_logger.info("End Training - logger.write()")


if __name__ == "__main__":
    # Based on https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system
    try:
        subprocess.check_output('nvidia-smi')
        loguru_logger.info("a GPU is connected.")
    except Exception:
        # TPU or CPU
        if "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]:
            import jax.tools.colab_tpu

            jax.tools.colab_tpu.setup_tpu()
            loguru_logger.info("A TPU is connected.")
        else:
            loguru_logger.info("Only CPU accelerator is connected.")


    # os.makedirs("configs", exist_ok=True)
    # config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
    # download_file(config_url, "configs/config.yaml")
    # env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env}.yaml"
    # os.makedirs("configs/env", exist_ok=True)
    # download_file(env_url, f"configs/env/{env}.yaml")

    with initialize(version_base=None, config_path="configs"):
        # cfg = compose(config_name="config.yaml", overrides=[f"env={env}", f"agent={agent}", "logger.type=terminal", "logger.save_checkpoint=true"])
        cfg = compose(config_name="config.yaml", overrides=[f"env={env}", f"agent={agent}", "logger.type=tensorboard", "logger.save_checkpoint=true"])

    train(cfg)
