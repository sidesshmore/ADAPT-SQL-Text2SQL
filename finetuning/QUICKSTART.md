# ADAPT-SQL Quick Start Guide - SOL Supercomputer (ASU)

Complete guide from initial setup to fine-tuning on ASU's SOL supercomputer.

---

## Part 1: Baseline System Setup & Testing

### Step 1: Initial SOL Connection

```bash
# SSH into SOL from your local machine
ssh <your_netid>@sol.asu.edu

# Request GPU node with A100 (for both baseline and fine-tuning)
salloc --nodes=1 --cpus-per-task=30 --mem=100G --gres=gpu:a100:1 --time=10:00:00

# Wait for allocation... you'll see:
# salloc: Granted job allocation XXXXX
# salloc: Waiting for resource configuration
# salloc: Nodes gpu-node-XXX are ready for job

# Note: You're now on the GPU node
```

**What this does**: Allocates a GPU node with 30 CPUs, 100GB RAM, 1 A100 GPU for 10 hours.

### Step 2: Clone Repository

```bash
# Navigate to your home directory
cd ~

# Clone the repository (replace with your actual repo URL)
git clone https://github.com/<your-username>/ADAPT-SQL.git

# Or if already cloned, pull latest changes
cd ADAPT-SQL
git pull origin main
```

### Step 3: Load Required Modules

```bash
# Load Python 3.10
module load python/3.10

# Load CUDA (required for GPU operations)
module load cuda/12.1

# Verify modules loaded
module list
# Should show python/3.10 and cuda/12.1
```

**What this does**: Loads system-level dependencies for Python and GPU computing.

### Step 4: Create Virtual Environment (Baseline)

```bash
# Make sure you're in the project root
cd ~/ADAPT-SQL

# Create virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Your prompt should now show (venv) prefix
# Example: (venv) [netid@gpu-node-042 ADAPT-SQL]$
```

### Step 5: Install Baseline Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# This will install:
# - streamlit (UI framework)
# - faiss-cpu (vector search)
# - ollama (LLM interface)
# - pandas, numpy, sqlparse, etc.

# Expected time: ~5-10 minutes
```

### Step 6: Install Ollama Models

```bash
# Pull the base code generation model
ollama pull qwen3-coder

# Expected output:
# pulling manifest
# pulling 8b85...
# ...
# success

# Pull the embedding model for vector search
ollama pull nomic-embed-text

# Expected time: ~10-15 minutes depending on network
```

**What this does**: Downloads pre-trained models to SOL for local inference.

### Step 7: Build Vector Store (CRITICAL)

```bash
# Make sure you're in project root with venv activated
cd ~/ADAPT-SQL
source venv/bin/activate

# Build FAISS index from Spider training data
python utils/vector_store.py

# Expected output:
# Loading Spider training data...
# Loaded 7000 training examples
# Generating embeddings...
# Progress: 1000/7000
# Progress: 2000/7000
# ...
# Saving vector store...
# Vector store saved successfully!

# Expected time: ~5-10 minutes
```

**What this creates**:
- `vector_store/spider_train.index` - FAISS index file
- `vector_store/examples.json` - Metadata for examples

**Why this is needed**: Step 4 of the pipeline uses this for example retrieval.

### Step 8: Verify Database Files

```bash
# Check Spider dev dataset exists
ls data/spider/dev.json

# Check databases directory
ls data/spider/spider_data/database/

# Should show directories like:
# academic/ activity_1/ aircraft/ ...

# Check a sample database file
ls data/spider/spider_data/database/concert_singer/
# Should show: concert_singer.sqlite
```

### Step 9: Test Baseline System - Interactive UI

```bash
# Make sure venv is activated
source venv/bin/activate

# Start the main Streamlit UI
streamlit run ui/app.py

# Expected output:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://10.x.x.x:8501
```

**Access the UI**:
1. If using VSCode Remote: Click the local URL
2. If using SSH tunnel:
   ```bash
   # On your LOCAL machine, run:
   ssh -L 8501:localhost:8501 <netid>@sol.asu.edu
   # Then open http://localhost:8501 in browser
   ```

**Test a query**:
1. Select database: "concert_singer"
2. Enter question: "How many singers are there?"
3. Click "Generate SQL"
4. Should see: `SELECT COUNT(*) FROM singer`

### Step 10: Test Batch Processing UI

```bash
# Stop the previous streamlit (Ctrl+C)

# Start batch processing UI
streamlit run ui/pages/batch_processing.py

# Access at http://localhost:8501
```

**Run a quick test**:
1. Click "Load Dataset"
2. Select `data/spider/dev.json`
3. Set "Number of examples" to 10 (for quick test)
4. Click "Run Batch Processing"
5. Monitor progress bar
6. Results will save to `RESULTS/` directory

### Step 11: Test Multi-Model Comparison UI

```bash
# Stop previous streamlit (Ctrl+C)

# Start multi-model UI
streamlit run ui/pages/multimodel.py
```

**What this allows**: Compare different models side-by-side on the same queries.

---

## Part 2: Full Baseline Evaluation (Optional)

### Run Complete Dev Set Evaluation

```bash
# This evaluates all 1,034 examples from Spider dev set
# Expected time: 2-3 hours

# Start batch processing UI
streamlit run ui/pages/batch_processing.py

# In UI:
# 1. Load data/spider/dev.json
# 2. Keep all examples (1034)
# 3. Model: qwen3-coder
# 4. Enable all options:
#    ✓ Enable Structural Reranking
#    ✓ Enable Normalization
#    ✓ Enable Retry on Validation Failure
# 5. Click "Run Batch Processing"
```

**Expected Baseline Results**:
- Execution Accuracy (EX): ~91.8%
- Exact Match (EM): ~35.0%
- Results saved to: `RESULTS/batch_results_<timestamp>.csv`

---

## Part 3: Fine-Tuning Setup

### Step 12: Create Fine-Tuning Virtual Environment

```bash
# Make sure you're on GPU node
cd ~/ADAPT-SQL

# Create separate venv for fine-tuning (to avoid conflicts)
python -m venv venv_finetune

# Activate it
source venv_finetune/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Why separate venv?**: Fine-tuning requires additional heavy dependencies (PyTorch, Transformers, Unsloth).

### Step 13: Install Fine-Tuning Dependencies

```bash
# Install all fine-tuning requirements
pip install -r finetuning/requirements_finetuning.txt

# This installs:
# - torch==2.1.0+cu121 (PyTorch with CUDA)
# - transformers>=4.37.0 (Hugging Face)
# - datasets (data loading)
# - peft (LoRA adapters)
# - unsloth (optimized fine-tuning)
# - accelerate, bitsandbytes, etc.

# Expected time: 15-20 minutes
# Expected size: ~8GB download
```

### Step 14: Verify GPU Access

```bash
# Check GPU is visible
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.1   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |   0  NVIDIA A100-SXM...  Off  | 00000000:07:00.0 Off |                    0 |
# | N/A   30C    P0    53W / 400W |      0MiB / 40960MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# If you see "No devices were found", you're not on a GPU node!
```

---

## Part 4: Fine-Tuning Pipeline

### Step 15: Prepare Training Data

```bash
# Activate fine-tuning venv
cd ~/ADAPT-SQL
source venv_finetune/bin/activate

# Run data preparation script
python finetuning/prepare_training_data.py

# Expected output:
# Loading Spider training data from data/spider/spider_data/train_spider.json...
# Loaded 7000 training examples
# Loading schemas...
# Converting to instruction format...
# Progress: 1000/7000
# Progress: 2000/7000
# ...
# Splitting into train/val (97/3 split)...
# Saved 6790 training examples to finetuning/train_data.jsonl
# Saved 210 validation examples to finetuning/val_data.jsonl
# Done!

# Expected time: 5-10 minutes
```

**What this creates**:
- `finetuning/train_data.jsonl` - ~6,790 training examples
- `finetuning/val_data.jsonl` - ~210 validation examples

**Data format** (each line in JSONL):
```json
{
  "instruction": "Generate SQL for: Show all students older than 18\n\nSchema:\ntable students: id (INTEGER), name (TEXT), age (INTEGER)",
  "output": "SELECT * FROM students WHERE age > 18"
}
```

### Step 16: Start Fine-Tuning

```bash
# Ensure GPU is available
nvidia-smi

# Start fine-tuning (this will take 6-8 hours)
python finetuning/train_qwen.py

# Expected output:
# Loading qwen3-coder (Qwen3-Coder-7B-Instruct)...
# Applying LoRA configuration...
# Loading training data...
# Loaded 6790 training examples
# Loaded 210 validation examples
# Starting training...
#
# Epoch 1/3:
# Step 10/1697 | Loss: 1.234 | LR: 0.0002
# Step 20/1697 | Loss: 1.123 | LR: 0.0002
# ...
# Step 100/1697 | Loss: 0.876 | Running validation...
# Validation Loss: 0.823
# Saved checkpoint to finetuning/checkpoints/checkpoint-100
# ...

# Expected time: 6-8 hours on A100
```

**Training Configuration**:
- Model: qwen3-coder (Qwen3-Coder-7B-Instruct, 7 billion parameters)
- Method: LoRA (4-bit quantization)
- Epochs: 3
- Batch size: 4 per GPU
- Gradient accumulation: 4 steps (effective batch size: 16)
- Learning rate: 2e-4
- Max sequence length: 2048 tokens

**Checkpoints saved to**:
- `finetuning/checkpoints/checkpoint-100/`
- `finetuning/checkpoints/checkpoint-200/`
- ...
- `finetuning/checkpoints/final_model/` (LoRA adapters only)
- `finetuning/checkpoints/merged_model/` (Full model for Ollama)

**Monitoring tips**:
- Training loss should decrease from ~1.2 to ~0.3
- Validation loss should follow similar trend
- If loss increases or plateaus, training may need adjustment

### Step 17: Monitor Training (Optional)

```bash
# In a separate terminal, watch the checkpoint directory
watch -n 60 ls -lh finetuning/checkpoints/

# Or monitor GPU usage
watch -n 5 nvidia-smi
```

### Step 18: Handle Training Interruption (If Needed)

```bash
# If training is interrupted, resume from last checkpoint
# Edit finetuning/train_qwen.py and add:
# resume_from_checkpoint="finetuning/checkpoints/checkpoint-XXX"

# Then re-run
python finetuning/train_qwen.py
```

---

## Part 5: Deploy Fine-Tuned Model

### Step 19: Transfer Model to Local Machine

```bash
# On YOUR LOCAL MACHINE (not on SOL):

# Create checkpoints directory
mkdir -p finetuning/checkpoints

# Transfer the merged model (~14GB)
scp -r <your_netid>@sol.asu.edu:~/ADAPT-SQL/finetuning/checkpoints/merged_model ./finetuning/checkpoints/

# Expected time: 20-30 minutes depending on network
# Progress will show:
# adapter_config.json           100%  234    23.4KB/s   00:00
# config.json                   100% 1.2KB  120.3KB/s   00:00
# model-00001-of-00004.safetensors  25%  3.5GB  15.2MB/s   08:12 ETA
# ...
```

### Step 20: Convert to Ollama Format (Local Machine)

```bash
# On YOUR LOCAL MACHINE
cd ADAPT-SQL

# Activate your local baseline venv
source venv/bin/activate

# Make sure Ollama is installed
which ollama
# If not: brew install ollama (macOS) or see https://ollama.ai

# Run conversion script
python finetuning/convert_to_ollama.py

# Expected output:
# Loading model from finetuning/checkpoints/merged_model...
# Creating Modelfile...
# Importing to Ollama as 'qwen3-spider-sql'...
# Success! Model available as: qwen3-spider-sql

# Expected time: 5 minutes
```

**What this creates**:
- Ollama model named `qwen3-spider-sql`
- Available system-wide via `ollama run qwen3-spider-sql`

### Step 21: Test Fine-Tuned Model

```bash
# Quick interactive test
ollama run qwen3-spider-sql

# In the Ollama prompt, try:
>>> Generate SQL: Show all students
# Should respond with SQL

# Exit with /bye

# Test via Python
python
```

```python
import ollama

response = ollama.chat(
    model='qwen3-spider-sql',
    messages=[{
        'role': 'user',
        'content': '''Generate SQL for: How many singers are there?

Schema:
table singer: singer_id (INTEGER), name (TEXT), age (INTEGER)'''
    }]
)

print(response['message']['content'])
# Expected: SELECT COUNT(*) FROM singer
```

---

## Part 6: Evaluate Fine-Tuned Model

### Step 22: Run Batch Evaluation with Fine-Tuned Model

```bash
# Start batch processing UI
streamlit run ui/pages/batch_processing.py
```

**In the UI**:
1. Click "Load Dataset" → Select `data/spider/dev.json`
2. **Model**: Select `qwen3-spider-sql` (your fine-tuned model)
3. Number of examples: 1034 (full dev set)
4. Enable all options:
   - ✓ Enable Structural Reranking
   - ✓ Enable Normalization
   - ✓ Enable Retry on Validation Failure
5. Click "Run Batch Processing"

**Expected time**: 2-3 hours for 1,034 examples

**Expected results**:
- Execution Accuracy (EX): 93-95% (↑1-3% from baseline)
- Exact Match (EM): 40-45% (↑5-10% from baseline)
- Reduced retry counts
- Better handling of complex queries

### Step 23: Compare Baseline vs Fine-Tuned

```bash
# Use multi-model comparison UI
streamlit run ui/pages/multimodel.py
```

**In the UI**:
1. Select databases to test
2. Enter test questions
3. Compare outputs from:
   - Model 1: `qwen3-coder` (baseline)
   - Model 2: `qwen3-spider-sql` (fine-tuned)
4. Analyze differences in:
   - Generated SQL quality
   - Retry attempts needed
   - Execution success rate

---

## Complete Command Summary

### One-Time Setup (Do Once)

```bash
# === ON SOL ===
ssh <netid>@sol.asu.edu
salloc --nodes=1 --cpus-per-task=30 --mem=100G --gres=gpu:a100:1 --time=10:00:00

cd ~
git clone <repo-url> ADAPT-SQL
cd ADAPT-SQL

module load python/3.10
module load cuda/12.1

# Baseline venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install Ollama models
ollama pull qwen3-coder
ollama pull nomic-embed-text

# Build vector store
python utils/vector_store.py

# Fine-tuning venv
python -m venv venv_finetune
source venv_finetune/bin/activate
pip install --upgrade pip
pip install -r finetuning/requirements_finetuning.txt
```

### Running Baseline System

```bash
# === ON SOL ===
cd ~/ADAPT-SQL
source venv/bin/activate

# Interactive UI
streamlit run ui/app.py

# Batch processing
streamlit run ui/pages/batch_processing.py

# Multi-model comparison
streamlit run ui/pages/multimodel.py
```

### Fine-Tuning Pipeline

```bash
# === ON SOL ===
cd ~/ADAPT-SQL
source venv_finetune/bin/activate

# Prepare data
python finetuning/prepare_training_data.py

# Train (6-8 hours)
python finetuning/train_qwen.py

# === ON LOCAL MACHINE ===
# Transfer model
scp -r <netid>@sol.asu.edu:~/ADAPT-SQL/finetuning/checkpoints/merged_model ./finetuning/checkpoints/

# Convert to Ollama
cd ADAPT-SQL
source venv/bin/activate
python finetuning/convert_to_ollama.py

# Test
ollama run qwen3-spider-sql

# Evaluate
streamlit run ui/pages/batch_processing.py
# Select model: qwen3-spider-sql
```

---

## Automated Pipeline (Alternative)

If you want to run data preparation + training in one go:

```bash
# On SOL GPU node
cd ~/ADAPT-SQL
source venv_finetune/bin/activate

# Make script executable (first time only)
chmod +x finetuning/run_complete_pipeline.sh

# Run complete pipeline
./finetuning/run_complete_pipeline.sh

# This will:
# 1. Check for GPU
# 2. Prepare training data
# 3. Start fine-tuning
# 4. Merge and save final model
```

---

## Troubleshooting

### "No GPU detected"
```bash
nvidia-smi
# If fails, you're not on GPU node. Request new allocation:
salloc --nodes=1 --cpus-per-task=30 --mem=100G --gres=gpu:a100:1 --time=10:00:00
```

### "CUDA out of memory"
```python
# Edit finetuning/train_qwen.py:
BATCH_SIZE = 2  # Reduce from 4
GRADIENT_ACCUMULATION_STEPS = 8  # Increase from 4
```

### "Ollama model not found"
```bash
ollama list  # Check available models
ollama pull qwen3-coder  # Re-download if missing
```

### "Vector store not found"
```bash
cd ~/ADAPT-SQL
source venv/bin/activate
python utils/vector_store.py  # Rebuild
```

### "Streamlit port already in use"
```bash
# Kill existing streamlit
pkill -f streamlit
# Or use different port
streamlit run ui/app.py --server.port 8502
```

### Training very slow
```python
# Quick test with subset:
# Edit finetuning/train_qwen.py:
NUM_EPOCHS = 1
MAX_STEPS = 100  # Add this line
```

---

## Expected Timeline

| Task | Duration | Location |
|------|----------|----------|
| Initial SOL setup | 30 mins | SOL |
| Install baseline dependencies | 10 mins | SOL |
| Download Ollama models | 15 mins | SOL |
| Build vector store | 10 mins | SOL |
| Test baseline UI | 10 mins | SOL |
| Install fine-tuning deps | 20 mins | SOL |
| Prepare training data | 10 mins | SOL |
| **Fine-tune model** | **6-8 hours** | **SOL** |
| Transfer model to local | 30 mins | Local |
| Convert to Ollama | 5 mins | Local |
| Evaluate fine-tuned model | 2-3 hours | Local |
| **TOTAL** | **~10-13 hours** | - |

---

## Next Steps After Fine-Tuning

1. **Analyze Results**:
   - Compare EX/EM metrics
   - Identify query types where fine-tuning helps most
   - Check retry count reduction

2. **Ablation Studies**:
   - Test without retry mechanism
   - Test without structural reranking
   - Direct generation only (bypass pipeline steps)

3. **Error Analysis**:
   - Examine failed queries
   - Compare error patterns: baseline vs fine-tuned
   - Identify remaining challenges

4. **Further Improvements**:
   - Try different epochs/learning rates
   - Fine-tune on additional datasets (WikiSQL, CoSQL)
   - Experiment with larger models (14B, 32B)

---

## Resources

- **ADAPT-SQL Documentation**: See `CLAUDE.md` in project root
- **Fine-Tuning Details**: See `finetuning/README_FINETUNING.md`
- **Spider Benchmark**: https://yale-lily.github.io/spider
- **Qwen3-Coder**: https://github.com/QwenLM/Qwen2.5-Coder (Qwen3 series)
- **Unsloth**: https://github.com/unslothai/unsloth
- **SOL Documentation**: https://asurc.atlassian.net/wiki/spaces/RC/overview
