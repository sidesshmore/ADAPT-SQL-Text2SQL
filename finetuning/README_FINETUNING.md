# Fine-Tuning Qwen3-Coder on Spider Dataset

This directory contains scripts for fine-tuning Qwen3-Coder on the Spider Text-to-SQL dataset using ASU's SOL supercomputer.

## Overview

**Goal**: Fine-tune Qwen3-Coder on direct (question, schema) → SQL pairs from Spider training data, then use the fine-tuned model in the ADAPT-SQL pipeline.

**Expected Outcome**: Compare performance of fine-tuned model vs. original qwen3-coder on Spider dev/test sets.

## Files

- `prepare_training_data.py` - Convert Spider JSON to instruction-tuning format
- `train_qwen.py` - Fine-tune Qwen3-Coder using Unsloth (A100 optimized)
- `convert_to_ollama.py` - Export fine-tuned model to Ollama format
- `requirements_finetuning.txt` - Python dependencies for fine-tuning

## SOL Supercomputer Setup

### 1. Start VSCode Session on SOL

```bash
# SSH into SOL
ssh <your_netid>@sol.asu.edu

# Request VSCode session with A100 GPU
# In SOL portal or via SLURM:
salloc --nodes=1 --cpus-per-task=30 --mem=100G --gres=gpu:a100:1 --time=10:00:00

# Note your allocated node (e.g., gpu-node-042)
# Connect VSCode to the node via SSH
```

### 2. Clone Project to SOL

```bash
# On SOL compute node
cd ~
git clone <your-repo-url> ADAPT-SQL
cd ADAPT-SQL
```

### 3. Setup Python Environment

```bash
# Load required modules
module load python/3.10
module load cuda/12.1

# Create virtual environment
python -m venv venv_finetune
source venv_finetune/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r finetuning/requirements_finetuning.txt
```

**Note**: Installation may take 15-20 minutes due to PyTorch and transformers packages.

### 4. Prepare Training Data

```bash
# Activate environment
source venv_finetune/bin/activate

# Run data preparation
python finetuning/prepare_training_data.py
```

**Output**:
- `finetuning/train_data.jsonl` - ~7,000 training examples
- `finetuning/val_data.jsonl` - 200 validation examples

**Expected time**: ~5-10 minutes

### 5. Fine-Tune Model

```bash
# Ensure you're on GPU node with A100
nvidia-smi  # Should show A100 GPU

# Start fine-tuning
python finetuning/train_qwen.py
```

**Training Configuration**:
- Model: Qwen2.5-Coder-7B-Instruct (via Unsloth)
- Method: LoRA (Low-Rank Adaptation) with 4-bit quantization
- Epochs: 3
- Batch size: 4 (effective: 16 with gradient accumulation)
- Learning rate: 2e-4
- Sequence length: 2048 tokens

**Expected time**: ~6-8 hours on A100

**Outputs**:
- `finetuning/checkpoints/final_model/` - LoRA adapters
- `finetuning/checkpoints/merged_model/` - Full merged model for Ollama

**Monitoring**:
- Training logs print every 10 steps
- Validation runs every 100 steps
- Checkpoints saved every 100 steps

### 6. Transfer Model to Local Machine

After training completes, transfer the merged model to your local machine:

```bash
# On your local machine
scp -r <netid>@sol.asu.edu:~/ADAPT-SQL/finetuning/checkpoints/merged_model ./finetuning/checkpoints/
```

**Expected transfer time**: ~20-30 minutes (model is ~14GB)

### 7. Convert to Ollama Format (Local Machine)

```bash
# On your local machine
cd ADAPT-SQL
source venv/bin/activate  # Your original venv

# Install Ollama if not already installed
# https://ollama.ai/download

# Convert and import
python finetuning/convert_to_ollama.py
```

**This creates**: Ollama model named `qwen3-spider-sql`

**Test the model**:
```bash
ollama run qwen3-spider-sql
```

### 8. Update Pipeline to Use Fine-Tuned Model

In your code, simply change the model parameter:

```python
# Before (using pre-trained model)
adapt = ADAPTBaseline(model="qwen3-coder")

# After (using fine-tuned model)
adapt = ADAPTBaseline(model="qwen3-spider-sql")
```

### 9. Run Batch Evaluation

```bash
# Start Streamlit UI
streamlit run ui/pages/batch_processing.py

# In the UI:
# 1. Select "qwen3-spider-sql" as model
# 2. Load data/spider/dev.json
# 3. Run batch processing
# 4. Compare results with baseline qwen3-coder
```

## Expected Results

### Baseline (qwen3-coder with 11-step pipeline)
- Execution Accuracy: 91.8%
- Exact Match: 35.0%

### Fine-Tuned Model (expected improvements)
- **Hypothesis**: Fine-tuning should improve:
  - Schema-specific understanding (table/column names)
  - Domain-specific SQL patterns from Spider
  - Potentially reduce need for complex retry logic

- **Expected EX**: 93-95% (↑1-3%)
- **Expected EM**: 40-45% (↑5-10%)

### If Fine-Tuned Model Performs Worse
This could indicate:
1. Overfitting to training set
2. Need for more training data augmentation
3. Hyperparameter tuning required
4. Original pipeline strategies were more effective

## Training Tips

### Reducing Training Time
If 6-8 hours is too long:
```python
# In train_qwen.py, modify:
NUM_EPOCHS = 1  # Instead of 3
MAX_STEPS = 1000  # Limit total steps
```

### Reducing Memory Usage
If you encounter OOM errors:
```python
# In train_qwen.py, modify:
BATCH_SIZE = 2  # Instead of 4
GRADIENT_ACCUMULATION_STEPS = 8  # Maintain effective batch size
```

### Using Checkpoints
If training is interrupted:
```python
# In train_qwen.py, add to TrainingArguments:
resume_from_checkpoint="finetuning/checkpoints/checkpoint-XXX"
```

## Troubleshooting

### "No module named 'unsloth'"
```bash
pip install --upgrade unsloth
```

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in `train_qwen.py`
- Reduce `MAX_SEQ_LENGTH` to 1024

### "Database not found" during data preparation
- Ensure `data/spider/spider_data/database/` exists
- Check that database subdirectories contain `.sqlite` files

### Ollama import fails
- Try the GGUF conversion method (requires llama.cpp)
- Alternatively, use the model directly via transformers:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("finetuning/checkpoints/merged_model")
  ```

## Alternative: Quick Test on Subset

For faster experimentation (30 minutes instead of 8 hours):

```python
# In prepare_training_data.py:
train_examples = prepare_spider_data(
    max_examples=500  # Only use 500 examples
)

# In train_qwen.py:
NUM_EPOCHS = 1
MAX_STEPS = 100
```

This won't produce production-quality results but validates the workflow.

## Next Steps After Fine-Tuning

1. **Ablation Study**: Test fine-tuned model with different pipeline components disabled
   - Without retry mechanism
   - Without structural reranking
   - Direct SQL generation only

2. **Error Analysis**: Compare failure cases between baseline and fine-tuned

3. **Multi-Model Comparison**: Use `ui/pages/multimodel.py` to compare:
   - qwen3-coder (baseline)
   - qwen3-spider-sql (fine-tuned)
   - Other models (deepseek-coder, etc.)

4. **Further Fine-Tuning**: If results are good, consider:
   - Fine-tuning on multiple datasets (WikiSQL, CoSQL)
   - Fine-tuning for specific pipeline steps
   - Using larger models (14B or 32B variants)

## Resources

- Unsloth Docs: https://github.com/unslothai/unsloth
- Qwen2.5-Coder: https://github.com/QwenLM/Qwen2.5-Coder
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Spider Benchmark: https://yale-lily.github.io/spider
