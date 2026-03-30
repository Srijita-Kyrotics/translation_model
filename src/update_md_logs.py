import os
import datetime

LOG_FILE = "data/final/training.log"
MD_FILE = "training_logs.md"

def update_logs():
    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    last_lines = lines[-100:] if len(lines) > 100 else lines
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(MD_FILE, "w") as f:
        f.write(f"# 🕒 Training Logs (Updated: {now})\n\n")
        f.write("```text\n")
        f.writelines(last_lines)
        f.write("\n```\n")
        f.write("\n\n---\n*This file is updated automatically every 60 seconds.*")

if __name__ == "__main__":
    update_logs()
