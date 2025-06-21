from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline


if __name__ == '__main__':
    parser = ArgumentParser(prog='trainer')
    parser.add_argument('jsonl_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    
    dataset = load_dataset("json", data_files=args.jsonl_file, split="train")
    
    model_name = "Qwen/Qwen3-4B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True  # Use bitsandbytes for QLoRA (low memory!)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print([name for name, module in model.named_modules()])
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["Wqkv", "o_proj", "gate_proj", "down_proj", "up_proj"]  # Example for Qwen3-4B
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Debug: confirm LoRA layers
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        fp16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to="none"
    )
    
    def preprocess(example):
        # Gather all messages in order; format them as in your JSONL
        prompt = ""
        for msg in example["messages"]:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        # Let's say you want to train to generate the assistant's last message only:
        last_user = [m for m in example["messages"] if m["role"] == "user"][-1]["content"]
        last_assistant = [m for m in example["messages"] if m["role"] == "assistant"][-1]["content"]
        input_text = f"<|im_start|>user\n{last_user}<|im_end|>\n<|im_start|>assistant\n"
        target_text = last_assistant + "<|im_end|>"
        full_text = input_text + target_text
    
        tokenized = tokenizer(
    	    full_text,
    	    truncation=True,
    	    max_length=1024,
    	    padding="max_length",
    	    return_tensors=None,
        )
        # For language modeling, labels == input_ids (except we mask the prompt tokens)
        input_ids = tokenized["input_ids"]
        labels = [-100] * len(tokenizer(input_text)["input_ids"]) + tokenized["input_ids"][len(tokenizer(input_text)["input_ids"]):]
        labels = labels[:1024]  # pad or truncate to max_length
    
        tokenized["labels"] = labels
        return tokenized
    
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    result = pipe("What is NRR and how is it used?")
    print(result[0]['generated_text'])
