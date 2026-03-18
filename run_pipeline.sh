#!/bin/bash

# Exit on any error
set -e

PYTHON_EXE="/home/kyrotics/anaconda3/envs/bhasantar_env/bin/python3"
export PYTHONPATH=.

echo "--------------------------------------------------"
echo "🚀 Starting IndicTrans2 Methodology Pipeline"
echo "--------------------------------------------------"

# 0. Start or Resume Back-Translation
if [ -f bt_pid.txt ]; then
    BT_PID=$(cat bt_pid.txt | tr -d '\n')
    if [ -n "$BT_PID" ] && ps -p "$BT_PID" > /dev/null; then
        echo "⏳ Back-translation is already running."
    else
        echo "⚠️  Process not running. Resuming..."
        nohup $PYTHON_EXE src/backtranslate.py > backtranslate_log.txt 2>&1 &
        echo $! > bt_pid.txt
        BT_PID=$!
    fi
else
    echo "🚀 Starting Back-Translation..."
    nohup $PYTHON_EXE src/backtranslate.py > backtranslate_log.txt 2>&1 &
    echo $! > bt_pid.txt
    BT_PID=$!
fi

echo "⏳ Waiting for back-translation to complete (PID: $BT_PID)..."
while ps -p "$BT_PID" > /dev/null; do
    sleep 300
done
echo "✅ Back-translation completed."

# 2. Check if the back-translated corpus exists
BT_FILE="data/final/backtranslated_corpus.csv"
if [ ! -f "$BT_FILE" ]; then
    echo "❌ Error: Back-translated corpus not found at $BT_FILE."
    exit 1
fi

echo "--------------------------------------------------"
echo "📦 Step 1: Packaging Dataset (HF Format)"
echo "--------------------------------------------------"
$PYTHON_EXE src/prepare_dataset.py

echo "--------------------------------------------------"
echo "🏗️  Step 2: Starting Stage 1 Fine-Tuning (BN -> EN)"
echo "--------------------------------------------------"
$PYTHON_EXE src/finetune.py --direction bn-en

echo "--------------------------------------------------"
echo "🏗️  Step 2.5: Starting Stage 1 Fine-Tuning (EN -> BN)"
echo "--------------------------------------------------"
$PYTHON_EXE src/finetune.py --direction en-bn

echo "--------------------------------------------------"
echo "💎 Step 3: Stage 1 Complete (Both Directions)"
echo "--------------------------------------------------"
echo "✅ Bidirectional Stage 1 complete."

echo "--------------------------------------------------"
echo "🎉 Pipeline execution finished successfully."
echo "--------------------------------------------------"
