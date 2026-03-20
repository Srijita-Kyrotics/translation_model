#!/bin/bash
# Phase 2 Translation Daemon
export PYTHONPATH="/home/kyrotics/ML Projects/translation_model-main:/home/kyrotics/ML Projects/Bhasantar_Legal_General/rotary-indictrans2-en-indic-1B/IndicTransToolkit"
cd "/home/kyrotics/ML Projects/translation_model-main"

# Force delete old log and start fresh
rm -f /home/kyrotics/translation.log

# Run and detach
/home/kyrotics/anaconda3/envs/bhasantar_env/bin/python -u src/translate_self_log.py 2>&1
