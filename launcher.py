import subprocess
import os
import sys

# Define absolute paths to avoid space-in-path shell issues
PYTHON_BIN = "/home/kyrotics/anaconda3/envs/bhasantar_env/bin/python"
SCRIPT_PATH = "/home/kyrotics/ML Projects/translation_model-main/src/translate_all.py"
LOG_PATH = "/home/kyrotics/ML Projects/translation_model-main/translation.log"
INDIC_TOOLKIT = "/home/kyrotics/ML Projects/Bhasantar_Legal_General/rotary-indictrans2-en-indic-1B/IndicTransToolkit"

# Set up environment
env = os.environ.copy()
env["PYTHONPATH"] = f".:{INDIC_TOOLKIT}"
env["HF_DATASETS_OFFLINE"] = "1"
env["TRANSFORMERS_OFFLINE"] = "1"
env["HF_HUB_DISABLE_XET"] = "1"

# Clear old log
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

print(f"Launching translation. Logs will be at: {LOG_PATH}")

# Launch background process with direct file redirection
with open(LOG_PATH, "w") as log_file:
    # Use start_new_session=True to detach from current terminal
    process = subprocess.Popen(
        [PYTHON_BIN, "-u", SCRIPT_PATH],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

print(f"Started translation with PID: {process.pid}")
