#!/bin/bash
# Runs ON the RunPod pod. Sets up PufferLib + pendulum6 and builds the env.
set -ex

apt-get update -qq
apt-get install -y -qq clang libomp-dev ccache rsync > /dev/null

cd /workspace
if [ ! -d PufferLib ]; then
    git clone --depth 1 https://github.com/PufferAI/PufferLib.git
fi
cd PufferLib

# pendulum6 files are rsynced into place by the local machine before this runs

pip install -e . --no-build-isolation 2>&1 | tail -2

nvcc --version | tail -1
./build.sh pendulum6
python -c "from pufferlib import _C; print('C backend OK')"
echo "SETUP DONE"
