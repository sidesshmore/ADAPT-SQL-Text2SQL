"""
Convert fine-tuned Qwen model to Ollama format
Creates a Modelfile and imports into Ollama
"""

import subprocess
import tempfile
from pathlib import Path


def create_modelfile(
    model_path: str,
    model_name: str = "qwen3-spider-sql",
    temperature: float = 0.1,
    top_p: float = 0.9,
    system_prompt: str = None
) -> str:
    """Create Ollama Modelfile"""

    if system_prompt is None:
        system_prompt = """You are an expert SQL generator. Given a database schema and a natural language question, generate the correct SQL query.

Rules:
1. Only use tables and columns that exist in the schema
2. Use proper SQL syntax (SQLite dialect)
3. Include necessary JOINs based on foreign keys
4. Use appropriate WHERE, GROUP BY, HAVING, ORDER BY clauses
5. Return only the SQL query without explanation"""

    modelfile = f"""FROM {model_path}

# Model parameters
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# System prompt
SYSTEM \"\"\"
{system_prompt}
\"\"\"

# Template (Qwen chat template)
TEMPLATE \"\"\"
<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""

    return modelfile


def convert_to_gguf(model_path: str, output_path: str):
    """Convert HuggingFace model to GGUF format (required for Ollama)"""

    print("Converting model to GGUF format...")
    print("This requires llama.cpp's convert script.")
    print("\nIf you don't have llama.cpp installed:")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  cd llama.cpp && make")
    print()

    # Check if conversion script exists
    llamacpp_path = Path.home() / "llama.cpp"
    convert_script = llamacpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"ERROR: Conversion script not found at {convert_script}")
        print("\nManual conversion steps:")
        print(f"  1. cd {llamacpp_path}")
        print(f"  2. python convert_hf_to_gguf.py {model_path} --outfile {output_path}")
        return False

    try:
        cmd = [
            "python",
            str(convert_script),
            model_path,
            "--outfile", output_path,
            "--outtype", "f16"  # Use FP16 precision
        ]

        subprocess.run(cmd, check=True)
        print(f"✓ Model converted to GGUF: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR during conversion: {e}")
        return False


def import_to_ollama(gguf_path: str, model_name: str, modelfile_content: str):
    """Import model into Ollama"""

    print(f"\nImporting model to Ollama as '{model_name}'...")

    # Create temporary Modelfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
        # Replace model path in Modelfile with GGUF path
        modelfile_with_path = modelfile_content.replace(
            "FROM {model_path}",
            f"FROM {gguf_path}"
        )
        f.write(modelfile_with_path)
        modelfile_path = f.name

    try:
        # Import using ollama create
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        subprocess.run(cmd, check=True)

        print(f"\n{'='*60}")
        print(f"✓ Model successfully imported to Ollama!")
        print(f"Model name: {model_name}")
        print(f"\nTest it with:")
        print(f"  ollama run {model_name}")
        print(f"\nUse in your pipeline:")
        print(f"  adapt = ADAPTBaseline(model='{model_name}')")
        print(f"{'='*60}\n")

        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR during Ollama import: {e}")
        return False
    finally:
        # Clean up temp file
        Path(modelfile_path).unlink()


def main():
    """Main conversion workflow"""

    print("="*60)
    print("CONVERT FINE-TUNED MODEL TO OLLAMA")
    print("="*60 + "\n")

    # Paths - Use scratch directory where train_qwen.py saves checkpoints
    # UPDATE: Change 'smore123' to your actual NetID
    SCRATCH_DIR = Path("/scratch/smore123/ADAPT-SQL")
    MERGED_MODEL_PATH = SCRATCH_DIR / "finetuning" / "checkpoints" / "merged_model"

    # Save GGUF to scratch as well (it's large)
    GGUF_OUTPUT_PATH = SCRATCH_DIR / "finetuning" / "qwen3-spider-sql.gguf"
    MODEL_NAME = "qwen3-spider-sql"

    # Check if merged model exists
    if not MERGED_MODEL_PATH.exists():
        print(f"ERROR: Merged model not found at {MERGED_MODEL_PATH}")
        print("Please run train_qwen.py first to create the fine-tuned model.")
        return

    print(f"Model path: {MERGED_MODEL_PATH}")
    print(f"GGUF output: {GGUF_OUTPUT_PATH}")
    print(f"Ollama model name: {MODEL_NAME}\n")

    # Option 1: Direct import (if model is already in compatible format)
    print("Attempting direct import to Ollama...")
    print("(If this fails, we'll convert to GGUF first)\n")

    modelfile = create_modelfile(str(MERGED_MODEL_PATH), MODEL_NAME)

    try:
        # Try direct import
        import_success = import_to_ollama(str(MERGED_MODEL_PATH), MODEL_NAME, modelfile)

        if import_success:
            print("Direct import successful!")
            return

    except Exception as e:
        print(f"Direct import failed: {e}")
        print("\nTrying GGUF conversion method...\n")

    # Option 2: Convert to GGUF first
    if convert_to_gguf(str(MERGED_MODEL_PATH), str(GGUF_OUTPUT_PATH)):
        # Update modelfile to use GGUF path
        modelfile = create_modelfile(str(GGUF_OUTPUT_PATH), MODEL_NAME)
        import_to_ollama(str(GGUF_OUTPUT_PATH), MODEL_NAME, modelfile)


if __name__ == "__main__":
    main()
