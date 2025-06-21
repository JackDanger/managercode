import os
import gc
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from peft import LoraConfig, get_peft_model, PeftModel

def get_model_and_tokenizer(model_name, lora_config=None, lora_weights_path=None, load_in_4bit=True):
    """Load model and tokenizer, optionally with LoRA."""
    from transformers import BitsAndBytesConfig

    # Configure quantization properly
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Match model dtype
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if lora_config:
        model = get_peft_model(model, lora_config)
    if lora_weights_path and os.path.exists(lora_weights_path):
        model = PeftModel.from_pretrained(model, lora_weights_path)
    return model, tokenizer

def preprocess(example, tokenizer, max_length=1024):
    """
    Preprocess a single conversation to tokenized inputs/labels for CausalLM training.
    Uses Qwen3 chat template with full conversation context.
    """
    messages = example.get("messages", [])
    if not messages or len(messages) < 2:
        return {}  # Filter out incomplete conversations

    # Build full conversation with all turns
    conversation_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            conversation_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    if not conversation_parts:
        return {}

    full_text = "\n".join(conversation_parts)

    # Find the last assistant response to determine where labels should start
    last_assistant_idx = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant_idx = i

    if last_assistant_idx is None:
        return {}  # No assistant response found

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    input_ids = tokenized["input_ids"]

    # Calculate where the last assistant response begins
    # Build text up to last assistant response
    context_parts = []
    for i, msg in enumerate(messages[:last_assistant_idx]):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            context_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    context_parts.append(f"<|im_start|>assistant\n")

    context_text = "\n".join(context_parts)
    
    # Tokenize context with truncation to ensure it doesn't exceed max_length
    context_tokens = tokenizer(context_text, add_special_tokens=False, truncation=True, max_length=max_length)
    context_len = len(context_tokens["input_ids"])
    
    # Skip if context alone fills the entire sequence (no room for assistant response)
    if context_len >= max_length - 10:  # Leave at least 10 tokens for response
        return {}
    
    # Create labels array with proper length from the start
    labels = [-100] * len(input_ids)
    # Only supervise tokens after the context
    for i in range(context_len, len(input_ids)):
        labels[i] = input_ids[i]
    
    tokenized["labels"] = labels
    return tokenized

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train(args):
    # 1. Load dataset
    try:
        dataset = load_dataset("json", data_files=args.jsonl_file, split="train")
        print(f"Loaded {len(dataset)} examples from {args.jsonl_file}")
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {args.jsonl_file}: {e}")

    model_name = args.model_name
    output_dir = args.output_dir

    # 2. Setup LoRA config with configurable parameters
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],  # Common for Qwen/Llama models
    )

    # 3. Load model/tokenizer (in 4bit quantization)
    model, tokenizer = get_model_and_tokenizer(model_name, lora_config=lora_config, load_in_4bit=True)

    # Prepare model for training
    model.config.use_cache = False  # Disable cache for training
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    # Clean memory before training
    clean_memory()

    # 4. Preprocess dataset
    def _preprocess(example):
        return preprocess(example, tokenizer, max_length=args.max_length)
    
    print(f"\nPreprocessing dataset with max_length={args.max_length}...")
    dataset = dataset.map(_preprocess, remove_columns=dataset.column_names)
    
    # Filter out empty examples
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: "input_ids" in x and x["input_ids"] is not None)
    filtered_len = len(dataset)
    
    if filtered_len < original_len:
        print(f"Filtered out {original_len - filtered_len} examples (likely due to context length constraints)")

    # Validate and show dataset info
    if len(dataset) == 0:
        raise ValueError("No valid examples found after preprocessing. Check your data format or increase max_length.")

    print(f"\nDataset info after preprocessing:")
    print(f"- Number of examples: {len(dataset)}")
    print(f"- Features: {dataset.features}")

    # Show a sample if verbose
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample tokenized length: {len(sample['input_ids'])}")
        print(f"Number of supervised tokens: {sum(1 for label in sample['labels'] if label != -100)}")
        
        # Additional validation
        assert len(sample['input_ids']) == len(sample['labels']), "Input and label lengths mismatch"
        assert len(sample['input_ids']) <= args.max_length, f"Sample exceeds max_length: {len(sample['input_ids'])} > {args.max_length}"

    # Clean memory before training
    clean_memory()

    # 5. Training args
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        remove_unused_columns=False,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",  # Memory efficient optimizer
        gradient_checkpointing=True,
    )

    # 6. Train!
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    # Save LoRA weights
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model & adapter saved to {output_dir}")

    # Clean up
    del trainer
    del dataset
    clean_memory()

def run_inference(args):
    model_name = args.model_name
    adapter_dir = args.output_dir
    prompt = args.prompt

    # 1. Load model + adapter
    print(f"Loading model {model_name} with adapter from {adapter_dir}")
    model, tokenizer = get_model_and_tokenizer(model_name, lora_weights_path=adapter_dir, load_in_4bit=True)

    # 2. Prepare prompt using chat template
    messages = [{"role": "user", "content": prompt}]
    user_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 3. Inference
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    result = pipe(
        user_prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=0.95,
        repetition_penalty=1.1
    )

    # Extract only the generated response
    generated_text = result[0]["generated_text"]
    response = generated_text[len(user_prompt):].strip()
    print("\n[Model Response]:\n", response)

    # Clean up
    del pipe
    clean_memory()

def parse_args():
    parser = argparse.ArgumentParser(prog="Fine-tuning Script",
                                     description="Fine-tune LLMs using LoRA with 4-bit quantization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run in training mode.")
    group.add_argument("--inference", action="store_true", help="Run in inference mode.")

    # Data and model arguments
    parser.add_argument("--jsonl_file", type=str, help="Path to training data (jsonl) [Required for train]")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save/load model or adapter.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Base model to use.")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (train mode).")
    parser.add_argument("--max_length", type=int, default=1024, help="Max tokens per sample (train mode).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    # Inference arguments
    parser.add_argument("--prompt", type=str, help="Prompt to run for inference mode.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for inference.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        if not args.jsonl_file:
            raise ValueError("You must provide --jsonl_file for training mode.")
        train(args)
    elif args.inference:
        if not args.prompt:
            raise ValueError("You must provide --prompt for inference mode.")
        run_inference(args)
