"""
Fine-tune Qwen3-Coder on Spider dataset using Unsloth
Optimized for A100 GPU with efficient training
"""

import json
import torch
import psutil  # Import psutil before unsloth to avoid cache issues
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


# Configuration
MAX_SEQ_LENGTH = 2048  # Qwen3-Coder supports up to 32k, but 2k is sufficient for most SQL
LOAD_IN_4BIT = True    # Use 4-bit quantization for memory efficiency
LORA_RANK = 16         # LoRA rank (higher = more parameters but better quality)
LORA_ALPHA = 16        # LoRA scaling factor
LORA_DROPOUT = 0.0     # Dropout for LoRA layers

# Training hyperparameters
BATCH_SIZE = 4         # Per device batch size (adjust based on memory)
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-4   # Learning rate for LoRA
NUM_EPOCHS = 3         # Number of training epochs
WARMUP_STEPS = 10      # Warmup steps
MAX_STEPS = -1         # -1 means use num_epochs instead

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "finetuning" / "train_data.jsonl"
VAL_DATA = PROJECT_ROOT / "finetuning" / "val_data.jsonl"

# Use scratch directory for checkpoints (more space on SOL supercomputer)
SCRATCH_DIR = Path("/scratch/smore123/ADAPT-SQL")
OUTPUT_DIR = SCRATCH_DIR / "finetuning" / "checkpoints"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_chat_template(example):
    """Format messages into Qwen chat template"""
    messages = example['messages']
    text = ""

    for msg in messages:
        role = msg['role']
        content = msg['content']

        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    return {"text": text}


def load_data():
    """Load and format training/validation data"""
    print("Loading datasets...")

    train_dataset = load_dataset('json', data_files=str(TRAIN_DATA), split='train')
    val_dataset = load_dataset('json', data_files=str(VAL_DATA), split='train')

    # Apply chat template formatting
    train_dataset = train_dataset.map(format_chat_template, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_chat_template, remove_columns=val_dataset.column_names)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_model():
    """Load and configure model with LoRA"""
    print("\nLoading Qwen3-Coder model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-Coder-7B-Instruct",  # Unsloth's optimized Qwen
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (will use bfloat16 on A100)
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("\nConfiguring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total params: {total_params:,}")

    return model, tokenizer


def train():
    """Main training loop"""
    print("="*60)
    print("QWEN3-CODER FINE-TUNING FOR SPIDER TEXT-TO-SQL")
    print("="*60)

    # Ensure psutil is available globally for unsloth's compiled cache
    import builtins
    builtins.psutil = psutil

    # Load data
    train_dataset, val_dataset = load_data()

    # Setup model
    model, tokenizer = setup_model()

    # Training arguments
    print("\nConfiguring training...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",  # Change to "wandb" if you want to use Weights & Biases
        dataloader_num_workers=4,  # Explicit worker count to avoid psutil auto-detection
        dataloader_pin_memory=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    trainer.train()

    # Save final model
    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)

    final_output_dir = OUTPUT_DIR / "final_model"
    model.save_pretrained(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    # Save merged model (LoRA weights merged into base model)
    print("\nSaving merged model for Ollama conversion...")
    merged_output_dir = OUTPUT_DIR / "merged_model"
    model.save_pretrained_merged(
        str(merged_output_dir),
        tokenizer,
        save_method="merged_16bit",  # or "merged_4bit" for smaller size
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final model saved to: {final_output_dir}")
    print(f"Merged model saved to: {merged_output_dir}")
    print("="*60 + "\n")

    return trainer


if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be very slow.")
        print("Make sure you're running this on the SOL supercomputer with GPU allocation.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    trainer = train()
