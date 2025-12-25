#!/bin/bash
# Complete fine-tuning pipeline for SOL supercomputer
# Run this script on the allocated GPU node

set -e  # Exit on error

echo "=========================================="
echo "ADAPT-SQL FINE-TUNING PIPELINE"
echo "=========================================="
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU node?"
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [ ! -d "venv_finetune" ]; then
    echo "ERROR: venv_finetune not found. Please run setup first:"
    echo "  python -m venv venv_finetune"
    echo "  source venv_finetune/bin/activate"
    echo "  pip install -r finetuning/requirements_finetuning.txt"
    exit 1
fi

source venv_finetune/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 1: Prepare training data
echo "=========================================="
echo "STEP 1: Preparing Training Data"
echo "=========================================="
if [ -f "finetuning/train_data.jsonl" ] && [ -f "finetuning/val_data.jsonl" ]; then
    echo "Training data already exists. Skip preparation? (y/n)"
    read -r skip_prep
    if [ "$skip_prep" != "y" ]; then
        python finetuning/prepare_training_data.py
    else
        echo "✓ Skipping data preparation"
    fi
else
    python finetuning/prepare_training_data.py
fi
echo ""

# Step 2: Fine-tune model
echo "=========================================="
echo "STEP 2: Fine-Tuning Model"
echo "=========================================="
echo "This will take approximately 6-8 hours on A100."
echo "Continue? (y/n)"
read -r continue_training

if [ "$continue_training" == "y" ]; then
    python finetuning/train_qwen.py
    echo ""
    echo "✓ Fine-tuning complete!"
else
    echo "Skipping fine-tuning."
    echo ""
fi

# Step 3: Summary
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Transfer model to your local machine:"
echo "   scp -r <netid>@sol.asu.edu:~/ADAPT-SQL/finetuning/checkpoints/merged_model ./finetuning/checkpoints/"
echo ""
echo "2. Convert to Ollama format (on local machine):"
echo "   python finetuning/convert_to_ollama.py"
echo ""
echo "3. Run batch evaluation:"
echo "   streamlit run ui/pages/batch_processing.py"
echo ""
echo "See finetuning/README_FINETUNING.md for detailed instructions."
echo ""
