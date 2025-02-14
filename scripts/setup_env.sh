#!/bin/bash

#
# This script should be run on a GPU node.
#

work_dir=${1:-$HOME/gmo-gpucloud-example}

echo "Setting up virtual environment"

if [ -d "$work_dir/.venv" ]; then
  rm -rf $work_dir/.venv
fi

source $work_dir/scripts/module_load.sh

python -m venv $work_dir/.venv
source $work_dir/.venv/bin/activate

pip install --upgrade pip

pip install -r $work_dir/requirements.txt

echo "Virtual environment setup complete"
