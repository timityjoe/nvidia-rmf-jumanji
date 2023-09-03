#!/bin/bash

# See https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html
echo "Setting up Isaac Gym Environment..."
source activate base	
conda deactivate
conda activate isaac-sim-2022-2-1
export ISAAC_SIM="/media/timityjoe/Data/Omniverse/share/ov/pkg/isaac_sim-2022.2.1"
source $ISAAC_SIM/setup_conda_env.sh
source $ISAAC_SIM/setup_python_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PYTHON_PATH=$ISAAC_SIM/python.sh
export EXP_PATH=$ISAAC_SIM/apps
export CARB_APP_PATH=$ISAAC_SIM/kit
export ISAAC_PATH=$ISAAC_SIM
echo "$LD_LIBRARY_PATH"
