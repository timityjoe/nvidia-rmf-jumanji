#!/bin/bash

echo "Setting up Instadeep Jumanji Environment..."
source activate base	
conda deactivate
conda activate conda39-jumanji
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
echo "$LD_LIBRARY_PATH"
