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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config={"load_in_4bit": load_in_4bit},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if lora_config:
        model = get_peft_model(model, lora_config)
    if lora_weights_path and os.path.exists(lora_weights_path):
        model = PeftModel.from_pretrained(model, lora_weights_path)
    return model, tokenizer

def preprocess(example, tokenizer, max_length=1024):
    """
    Preprocess a single conversation to tokenized inputs/labels for CausalLM training.
    Uses Qwen3 chat template: https://huggingface.co/Qwen/Qwen1.5-4B-Chat#how-to-use-the-chat-template
    """
    # Assemble as user-assistant final turn
    user_msgs = [m["content"] for m in example["messages"] if m["role"] == "user"]
    assistant_msgs = [m["content"] for m in example["messages"] if m["role"] == "assistant"]
    if not user_msgs or not assistant_msgs:
        return {}  # Filter out incomplete conversations

    # You can adjust to use all turns, but here's just the last user->assistant
    input_text = (
        "<|im_start|>user\n" + user_msgs[-1] + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    target_text = assistant_msgs[-1] + "<|im_end|>"

    full_text = input_text + target_text

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    input_ids = tokenized["input_ids"]
    prompt_len = len(tokenizer(input_text)["input_ids"])
    # Only supervise assistant response
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[:max_length]
    tokenized["labels"] = labels
    return tokenized

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train(args):
    # 1. Load dataset
    dataset = load_dataset("json", data_files=args.jsonl_file, split="train")
    model_name = args.model_name
    output_dir = args.output_dir

    # 2. Setup LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],  # Use model.named_modules() for precise names if needed
    )

    # 3. Load model/tokenizer (in 4bit quantization)
    model, tokenizer = get_model_and_tokenizer(model_name, lora_config=lora_config, load_in_4bit=True)
    model.print_trainable_parameters()

    # 4. Preprocess dataset
    def _preprocess(example):
        return preprocess(example, tokenizer, max_length=args.max_length)
    dataset = dataset.map(_preprocess, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: "input_ids" in x and x["input_ids"] is not None)

    # 5. Training args
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=args.epochs,
        fp16=True,
        remove_unused_columns=False,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
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
    model, tokenizer = get_model_and_tokenizer(model_name, lora_weights_path=adapter_dir, load_in_4bit=True)

    # 2. Prepare prompt using chat template if needed
    # Use Qwen3 chat template for best results
    user_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    # 3. Inference
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    result = pipe(user_prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    print("\n[Model output]:\n", result[0]["generated_text"])

    # Clean up
    del pipe
    clean_memory()

def parse_args():
    parser = argparse.ArgumentParser(prog="Qwen3 Slack Fine-tuner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run in training mode.")
    group.add_argument("--inference", action="store_true", help="Run in inference mode.")

    parser.add_argument("--jsonl_file", type=str, help="Path to training data (jsonl) [Required for train]")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save/load model or adapter.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Base model to use.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (train mode).")
    parser.add_argument("--max_length", type=int, default=1024, help="Max tokens per sample (train mode).")
    parser.add_argument("--prompt", type=str, help="Prompt to run for inference mode.")
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