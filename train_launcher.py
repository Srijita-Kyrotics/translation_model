import subprocess
import os
import time
import sys

# Paths from existing launcher.py
PYTHON_BIN = "/home/kyrotics/anaconda3/envs/bhasantar_env/bin/python"
FINETUNE_SCRIPT = "src/finetune.py"
LOG_PATH = "data/final/training.log"
UPDATE_SCRIPT = "src/update_md_logs.py"
INDIC_TOOLKIT = "/home/kyrotics/ML Projects/Bhasantar_Legal_General/rotary-indictrans2-en-indic-1B/IndicTransToolkit"

# Environment Setup
env = os.environ.copy()
env["PYTHONPATH"] = f".:{INDIC_TOOLKIT}"
env["WANDB_DISABLED"] = "true"  # Prevent interactive prompts
env["HF_DATASETS_OFFLINE"] = "0"
env["TRANSFORMERS_OFFLINE"] = "0"

# Prepare Directories
os.makedirs("data/final", exist_ok=True)

# Clear old log if it exists
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

print(f"🚀 Launching Stage 1 Training for en-bn...")
print(f"📝 Logs will be at: {LOG_PATH}")
print(f"📖 Viewable at: training_logs.md (Updated every 60s)")

# Step 1: Launch training in a subprocess
with open(LOG_PATH, "w") as log_file:
    train_process = subprocess.Popen(
        [PYTHON_BIN, "-u", FINETUNE_SCRIPT, "--direction", "en-bn"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

print(f"Process started with PID: {train_process.pid}")

# Step 2: Background Loop to update training_logs.md
try:
    while train_process.poll() is None:
        # Update logs.md
        subprocess.run([PYTHON_BIN, UPDATE_SCRIPT], env=env)
        time.sleep(60)
except KeyboardInterrupt:
    print("Launcher stopped manually.")
finally:
    if train_process.poll() is None:
        print(f"Training is still running in background (PID: {train_process.pid})")
    else:
        print("Training complete.")
