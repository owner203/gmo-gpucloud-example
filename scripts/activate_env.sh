#!/bin/bash

work_dir=${1:-$HOME/gmo-gpucloud-example}

source $work_dir/scripts/module_load.sh

if [ -d "$work_dir/.venv" ]; then
  source $work_dir/.venv/bin/activate
  echo "Virtual environment activated on $(hostname)"
else
  echo "Virtual environment not found. Please run 'sbatch setup_env.sbatch' first"
  exit 1
fi
