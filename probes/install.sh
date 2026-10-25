#!/bin/bash
# README "Other Linux instance" install. Usage: install.sh <pip-only|runtime>
set -e
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
export DEBIAN_FRONTEND=noninteractive
if [ "$1" = pip-only ]; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq wget gnupg python3 python3-venv binutils git >/dev/null
  python3 -m venv /tmp/venv
  /tmp/venv/bin/pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
  /tmp/venv/bin/pip install -q "torch-neuronx==2.14.*" "neuronx-cc==2.*" "nki>=0.6" --extra-index-url https://pip.repos.neuron.amazonaws.com
  /tmp/venv/bin/pip list 2>/dev/null | grep -E "^(torch|torch-neuronx|neuronx-cc|nki) "
else
  . /etc/os-release
  wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | $SUDO gpg --dearmor -o /usr/share/keyrings/neuron.gpg
  echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" | $SUDO tee /etc/apt/sources.list.d/neuron.list >/dev/null
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools >/dev/null
  ls /opt/aws/neuron/lib/libnrt.so.1
fi
