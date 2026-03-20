#!/bin/bash
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=".:/home/kyrotics/ML Projects/Bhasantar_Legal_General/rotary-indictrans2-en-indic-1B/IndicTransToolkit"

# Use absolute path to the verified environment's python
/home/kyrotics/anaconda3/envs/bhasantar_env/bin/python -u src/translate_all.py
