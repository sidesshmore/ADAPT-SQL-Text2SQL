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

### Step 6: Configure Ollama and Install Models

**IMPORTANT**: First configure Ollama to use scratch directory (to avoid running out of space):

```bash
# Configure Ollama to store models in scratch directory
# This is CRITICAL - home directory only has ~50GB, models need much more space
export OLLAMA_MODELS=/scratch/smore123/ADAPT-SQL/ollama_models

# Create the directory
mkdir -p /scratch/smore123/ADAPT-SQL/ollama_models

# Add to .bashrc to persist across sessions
echo 'export OLLAMA_MODELS=/scratch/smore123/ADAPT-SQL/ollama_models' >> ~/.bashrc

# Verify the setting
echo "OLLAMA_MODELS is now: $OLLAMA_MODELS"
```

**Note**: Replace `smore123` with your actual NetID in the paths above.

Now install the Ollama models:

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

**What this does**:
- Configures Ollama to use scratch directory (~10TB space) instead of home (~50GB)
- Downloads pre-trained models to SOL for local inference

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

**Important Note on Storage**:
- **All large files use scratch directory** (`/scratch/<netid>/ADAPT-SQL/`) to avoid space issues
- Training checkpoints: `/scratch/<netid>/ADAPT-SQL/finetuning/checkpoints/`
  - Configured in `train_qwen.py` (line 38-39): `SCRATCH_DIR = Path("/scratch/smore123/ADAPT-SQL")`
  - **Update the NetID** in `train_qwen.py` to match yours!
- Ollama models: `/scratch/<netid>/ADAPT-SQL/ollama_models/`
  - Configured via `OLLAMA_MODELS` environment variable (set in Step 6)
  - **CRITICAL**: Must be set before using Ollama to avoid "no space left on device" errors
- Scratch provides ~10TB space vs ~50GB in home directory
- Files persist but may be cleaned if unused for 60+ days

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
# IMPORTANT: First update the NetID in train_qwen.py
# Edit line 38 in train_qwen.py:
# Change: SCRATCH_DIR = Path("/scratch/smore123/ADAPT-SQL")
# To:     SCRATCH_DIR = Path("/scratch/<your_netid>/ADAPT-SQL")

# Quick way to update:
sed -i 's/smore123/<your_netid>/g' finetuning/train_qwen.py

# Or manually edit the file:
# nano finetuning/train_qwen.py  # or vim, emacs, etc.

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

**Checkpoints saved to** (in scratch directory):
- `/scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/checkpoint-100/`
- `/scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/checkpoint-200/`
- ...
- `/scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/final_model/` (LoRA adapters only)
- `/scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/merged_model/` (Full model for Ollama)

**Monitoring tips**:
- Training loss should decrease from ~1.2 to ~0.3
- Validation loss should follow similar trend
- If loss increases or plateaus, training may need adjustment

### Step 17: Monitor Training (Optional)

```bash
# In a separate terminal, watch the checkpoint directory in scratch
watch -n 60 ls -lh /scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/

# Example:
# watch -n 60 ls -lh /scratch/smore123/ADAPT-SQL/finetuning/checkpoints/

# Or monitor GPU usage
watch -n 5 nvidia-smi
```

### Step 18: Handle Training Interruption (If Needed)

```bash
# If training is interrupted, resume from last checkpoint
# First, find the latest checkpoint:
ls -lth /scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/

# Edit finetuning/train_qwen.py and add to training_args:
# resume_from_checkpoint="/scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/checkpoint-XXX"

# Then re-run
python finetuning/train_qwen.py
```

---

## Part 5: Deploy Fine-Tuned Model on SOL

### Step 19: Verify Fine-Tuned Model Checkpoints

```bash
# Make sure you're still on SOL GPU node
# Note: Checkpoints are saved to scratch directory for more space
# The path is defined in train_qwen.py: /scratch/<your_netid>/ADAPT-SQL/

# Check that training completed successfully
ls -lh /scratch/smore123/ADAPT-SQL/finetuning/checkpoints/merged_model/

# Should show:
# config.json
# generation_config.json
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# model-00003-of-00004.safetensors
# model-00004-of-00004.safetensors
# tokenizer.json
# tokenizer_config.json
# ...

# Total size should be ~14GB
```

**Important**: Replace `smore123` with your actual NetID in all paths below.

### Step 20: Create Modelfile for Ollama

```bash
# Create a Modelfile to import the fine-tuned model into Ollama
# Navigate to the merged model directory in scratch
cd /scratch/smore123/ADAPT-SQL/finetuning/checkpoints/merged_model

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./model-00001-of-00004.safetensors
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
SYSTEM """You are an expert SQL generator. Generate SQL queries based on natural language questions and database schemas. Only output the SQL query without explanations."""
EOF

# Verify Modelfile created
cat Modelfile
```

### Step 21: Import Model into Ollama

```bash
# Ensure OLLAMA_MODELS is set to scratch directory
# (Should be set from Step 6, but verify)
echo "OLLAMA_MODELS is: $OLLAMA_MODELS"
# Should show: /scratch/smore123/ADAPT-SQL/ollama_models

# If not set, configure it now:
# export OLLAMA_MODELS=/scratch/smore123/ADAPT-SQL/ollama_models

# Import the fine-tuned model into Ollama on SOL
ollama create qwen3-spider-sql -f Modelfile

# Expected output:
# transferring model data
# using existing layer sha256:xxxxx
# creating new layer sha256:yyyyy
# writing manifest
# success

# Verify model is available
ollama list

# Should show:
# NAME                    ID              SIZE      MODIFIED
# qwen3-spider-sql       xxxxx           14 GB     X seconds ago
# qwen3-coder            yyyyy           7 GB      X hours ago
# nomic-embed-text       zzzzz           274 MB    X hours ago

# Expected time: 5-10 minutes
```

**Note**: If the Modelfile approach doesn't work with safetensors, you may need to use the conversion script:

```bash
# Alternative: Use conversion script on SOL
cd ~/ADAPT-SQL
source venv/bin/activate

# The script will automatically look for the model in the scratch directory
# Make sure to update the SCRATCH_DIR path in convert_to_ollama.py if needed
python finetuning/convert_to_ollama.py

# This will create the Ollama model automatically
```

**Scratch Directory Notes**:
- All checkpoints are saved to `/scratch/<your_netid>/ADAPT-SQL/finetuning/checkpoints/`
- Ollama models are stored in `/scratch/<your_netid>/ADAPT-SQL/ollama_models/`
- Configured via:
  - `train_qwen.py` (line 38-39) - for training checkpoints
  - `OLLAMA_MODELS` environment variable - for Ollama models
- Scratch has more space than home directory (~10TB vs ~50GB)
- Files in scratch are persistent but may be cleaned if unused for 60+ days

### Step 22: Test Fine-Tuned Model on SOL

```bash
# Quick interactive test
ollama run qwen3-spider-sql

# In the Ollama prompt, try:
>>> Generate SQL for: How many singers are there?

Schema:
table singer: singer_id (INTEGER), name (TEXT), age (INTEGER)

# Should respond with: SELECT COUNT(*) FROM singer
# Exit with /bye or Ctrl+D

# Test via Python
python << 'PYEOF'
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
PYEOF

# Expected output: SELECT COUNT(*) FROM singer
```

---

## Part 6: Evaluate Fine-Tuned Model on SOL

### Step 23: Setup SSH Tunnel for Streamlit UI

Since you'll be running Streamlit on SOL, you need to access it from your local browser.

```bash
# On YOUR LOCAL MACHINE, open a new terminal and run:
ssh -L 8501:localhost:8501 <your_netid>@sol.asu.edu

# This creates a tunnel from your local port 8501 to SOL's port 8501
# Keep this terminal open while using the UI
```

### Step 24: Run Batch Evaluation with Fine-Tuned Model

```bash
# On SOL GPU node (or login node if evaluation doesn't need GPU)
cd ~/ADAPT-SQL
source venv/bin/activate

# Start batch processing UI
streamlit run ui/pages/batch_processing.py

# Expected output:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://10.x.x.x:8501
```

**Access the UI**:
1. Open your local browser to `http://localhost:8501` (using the SSH tunnel)
2. The Streamlit interface should load

**In the UI**:
1. Click "Load Dataset" → Select `data/spider/dev.json`
2. **Model**: Select `qwen3-spider-sql` (your fine-tuned model)
3. Number of examples:
   - Start with 50-100 for quick test
   - Then run full 1,034 for complete evaluation
4. Enable all options:
   - ✓ Enable Structural Reranking
   - ✓ Enable Normalization
   - ✓ Enable Retry on Validation Failure
5. Click "Run Batch Processing"

**Expected time**:
- 50 examples: ~15-20 minutes
- 1,034 examples (full dev set): 2-3 hours

**Expected results**:
- Execution Accuracy (EX): 93-95% (↑1-3% from baseline)
- Exact Match (EM): 40-45% (↑5-10% from baseline)
- Reduced retry counts
- Better handling of complex queries

**Results will be saved to**: `~/ADAPT-SQL/RESULTS/batch_results_<timestamp>.csv`

### Step 25: Compare Baseline vs Fine-Tuned

```bash
# Stop the previous Streamlit (Ctrl+C)

# Start multi-model comparison UI
streamlit run ui/pages/multimodel.py

# Access at http://localhost:8501 (via SSH tunnel)
```

**In the UI**:
1. Select databases to test (e.g., "concert_singer", "student_transcript")
2. Enter test questions
3. Compare outputs from:
   - Model 1: `qwen3-coder` (baseline)
   - Model 2: `qwen3-spider-sql` (fine-tuned)
4. Analyze differences in:
   - Generated SQL quality
   - Retry attempts needed
   - Execution success rate
   - Response time

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

# Configure Ollama to use scratch (CRITICAL - avoid space issues)
export OLLAMA_MODELS=/scratch/smore123/ADAPT-SQL/ollama_models  # Replace smore123 with your NetID
mkdir -p /scratch/smore123/ADAPT-SQL/ollama_models
echo 'export OLLAMA_MODELS=/scratch/smore123/ADAPT-SQL/ollama_models' >> ~/.bashrc

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

# Ensure OLLAMA_MODELS is set (should be in .bashrc from Step 6)
# Verify with:
echo $OLLAMA_MODELS
# If empty, run: export OLLAMA_MODELS=/scratch/<your_netid>/ADAPT-SQL/ollama_models

# Interactive UI
streamlit run ui/app.py

# Batch processing
streamlit run ui/pages/batch_processing.py

# Multi-model comparison
streamlit run ui/pages/multimodel.py
```

### Fine-Tuning Pipeline (All on SOL)

```bash
# === ON SOL ===
cd ~/ADAPT-SQL
source venv_finetune/bin/activate

# Prepare data
python finetuning/prepare_training_data.py

# Train (6-8 hours)
python finetuning/train_qwen.py

# Import fine-tuned model to Ollama on SOL
# Note: Replace smore123 with your NetID

# Ensure OLLAMA_MODELS is set (from Step 6)
echo "OLLAMA_MODELS is: $OLLAMA_MODELS"
# If not set: export OLLAMA_MODELS=/scratch/smore123/ADAPT-SQL/ollama_models

cd /scratch/smore123/ADAPT-SQL/finetuning/checkpoints/merged_model
cat > Modelfile << 'EOF'
FROM ./model-00001-of-00004.safetensors
PARAMETER temperature 0.1
PARAMETER top_p 0.9
SYSTEM """You are an expert SQL generator."""
EOF

ollama create qwen3-spider-sql -f Modelfile

# Or use the conversion script:
# cd ~/ADAPT-SQL && source venv/bin/activate && python finetuning/convert_to_ollama.py

# Test fine-tuned model
ollama run qwen3-spider-sql

# === ON LOCAL MACHINE (for accessing Streamlit) ===
# Setup SSH tunnel
ssh -L 8501:localhost:8501 <netid>@sol.asu.edu

# === BACK ON SOL ===
# Evaluate fine-tuned model
cd ~/ADAPT-SQL
source venv/bin/activate
streamlit run ui/pages/batch_processing.py
# Then open http://localhost:8501 in your local browser
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

### "No space left on device" (Ollama)
```bash
# This happens when Ollama tries to use home directory (~50GB) instead of scratch
# Solution: Configure OLLAMA_MODELS to use scratch

# Set the environment variable
export OLLAMA_MODELS=/scratch/<your_netid>/ADAPT-SQL/ollama_models
mkdir -p /scratch/<your_netid>/ADAPT-SQL/ollama_models

# Make it permanent
echo 'export OLLAMA_MODELS=/scratch/<your_netid>/ADAPT-SQL/ollama_models' >> ~/.bashrc

# Check available space
df -h /scratch/<your_netid>/  # Should show ~10TB
df -h ~  # Home directory, only ~50GB

# Clean up old ollama files from home (optional)
du -sh ~/.ollama  # Check current usage
rm -rf ~/.ollama/models  # Remove old models from home

# Retry the ollama command
```

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
| Import model to Ollama on SOL | 10 mins | SOL |
| Test fine-tuned model | 5 mins | SOL |
| Evaluate fine-tuned model | 2-3 hours | SOL |
| **TOTAL** | **~9-12 hours** | **SOL** |

---

## Accessing Results from SOL

Since all processing happens on SOL, you'll need to transfer result files to view them locally:

### Transfer Evaluation Results

```bash
# On YOUR LOCAL MACHINE
# Transfer results files
scp <netid>@sol.asu.edu:~/ADAPT-SQL/RESULTS/batch_results_*.csv ./RESULTS/
scp <netid>@sol.asu.edu:~/ADAPT-SQL/RESULTS/batch_results_*.pdf ./RESULTS/

# Or transfer entire results directory
scp -r <netid>@sol.asu.edu:~/ADAPT-SQL/RESULTS ./

# View CSV files locally
# open RESULTS/batch_results_<timestamp>.csv  # macOS
# Or use Excel, LibreOffice, etc.
```

### Transfer Model Checkpoints (Optional)

If you want to backup or use the fine-tuned model locally:

```bash
# On YOUR LOCAL MACHINE
# Transfer the merged model from scratch directory (~14GB, takes 20-30 min)
scp -r <netid>@sol.asu.edu:/scratch/<netid>/ADAPT-SQL/finetuning/checkpoints/merged_model ./finetuning/checkpoints/

# Example with actual NetID:
# scp -r smore123@sol.asu.edu:/scratch/smore123/ADAPT-SQL/finetuning/checkpoints/merged_model ./finetuning/checkpoints/
```

**Note**: Model checkpoints are stored in the scratch directory (`/scratch/<netid>/ADAPT-SQL/`) for space efficiency.

Alternatively, you can analyze results directly on SOL:

```bash
# On SOL
cd ~/ADAPT-SQL/RESULTS

# View CSV in terminal
column -t -s, batch_results_*.csv | less -S

# Or use Python for analysis
python << 'PYEOF'
import pandas as pd

# Load latest results
df = pd.read_csv('batch_results_<timestamp>.csv')

# Summary statistics
print(df['execution_match'].value_counts())
print(df['exact_match'].value_counts())
print(f"Execution Accuracy: {df['execution_match'].mean():.2%}")
print(f"Exact Match: {df['exact_match'].mean():.2%}")
PYEOF
```

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
