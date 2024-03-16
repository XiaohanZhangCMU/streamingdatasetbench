#!/bin/sh

echo "HELLO"

pip install -r requirements.txt
pip install --no-cache-dir git+https://github.com/Lightning-AI/pytorch-lightning.git@master
pip install -U lightning-cloud
pip install mosaicml
pip install litdata
