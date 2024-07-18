#!/bin/bash

module load stack/2024-04 gcc/8.5.0 python/3.9.18
python3 -m venv $HOME/maplett
source $HOME/maplett/bin/activate
pip install --upgrade pip setuptools
pip install -r $HOME/MA/requirements.txt
