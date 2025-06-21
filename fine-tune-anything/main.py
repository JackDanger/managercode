import os
import gc
import torch
import argparse
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    pipeline,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class ModelConfig:
    """Configuration for model loading and quantization."""

    name: str
    use_4bit_quantization: bool = True
    compute_dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True


@dataclass
class LoRAConfiguration:
    """Configuration for LoRA fine-tuning parameters."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ]


@dataclass
class TrainingConfiguration:
    """Configuration for training hyperparameters."""

    output_dir: str
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_sequence_length: int = 1024
    warmup_ratio: float = 0.1
    logging_steps: int = 20
    save_steps: int = 50
    save_total_limit: int = 2


@dataclass
class InferenceConfiguration:
    """Configuration for inference parameters."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1


class MemoryManager:
    """Manages GPU memory cleanup operations."""

    @staticmethod
    def cleanup():
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelManager:
    """Manages model loading, quantization, and adapter configuration."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration for 4-bit loading."""
        if not self.config.use_4bit_quantization:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.config.compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def load_base_model(self):
        """Load the base model and tokenizer."""
        quantization_config = self._create_quantization_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=self.config.compute_dtype,
            trust_remote_code=self.config.trust_remote_code,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name, trust_remote_code=self.config.trust_remote_code
        )

        self._ensure_padding_token()

    def _ensure_padding_token(self):
        """Ensure the tokenizer has a padding token set."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def apply_lora_adapter(self, lora_config: LoRAConfiguration):
        """Apply LoRA adapter to the model."""
        peft_config = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_config.target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)

    def load_trained_adapter(self, adapter_path: str):
        """Load a previously trained LoRA adapter."""
        if os.path.exists(adapter_path):
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def prepare_for_training(self):
        """Configure model for training."""
        self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()

    def save(self, output_dir: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class Message:
    """Represents a single message in a conversation."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def is_valid(self) -> bool:
        """Check if the message has both role and content."""
        return bool(self.role and self.content)

    def format(self) -> str:
        """Format message with chat template markers."""
        return f"<|im_start|>{self.role}\n{self.content}<|im_end|>"


class Conversation:
    """Represents a conversation consisting of multiple messages."""

    def __init__(self, messages: List[Dict[str, str]]):
        self.messages = [
            Message(msg.get("role", ""), msg.get("content", "")) for msg in messages
        ]

    def is_valid(self) -> bool:
        """Check if conversation has at least 2 valid messages."""
        valid_messages = [msg for msg in self.messages if msg.is_valid()]
        return len(valid_messages) >= 2

    def get_assistant_indices(self) -> List[int]:
        """Find indices of all assistant messages."""
        return [i for i, msg in enumerate(self.messages) if msg.role == "assistant"]

    def slice(self, start: int, end: int) -> "Conversation":
        """Create a sub-conversation from a slice of messages."""
        return Conversation(
            [
                {"role": msg.role, "content": msg.content}
                for msg in self.messages[start : end + 1]
            ]
        )

    def format_as_text(self) -> str:
        """Format entire conversation as text."""
        formatted_messages = [msg.format() for msg in self.messages if msg.is_valid()]
        return "\n".join(formatted_messages) if formatted_messages else ""


class ChunkBuilder:
    """Builds training chunks from conversations."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = 50  # Safety buffer for tokenization

    def build_from_conversation(
        self, conversation: Conversation, start_idx: int, end_idx: int
    ) -> Optional[Dict[str, List[int]]]:
        """Build a training chunk from a conversation slice."""
        chunk_conversation = conversation.slice(start_idx, end_idx)

        if not self._has_assistant_response(chunk_conversation):
            return None

        full_text = chunk_conversation.format_as_text()
        if not full_text:
            return None

        tokenized = self._tokenize_text(full_text)
        last_assistant_idx = self._find_last_assistant_index(chunk_conversation)

        context_length = self._calculate_context_length(
            chunk_conversation, last_assistant_idx
        )

        labels = self._create_training_labels(tokenized, context_length)
        tokenized["labels"] = labels

        return tokenized

    def _has_assistant_response(self, conversation: Conversation) -> bool:
        """Check if conversation contains at least one assistant response."""
        return any(msg.role == "assistant" for msg in conversation.messages)

    def _tokenize_text(self, text: str) -> Dict[str, List[int]]:
        """Tokenize text with padding and truncation."""
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

    def _find_last_assistant_index(self, conversation: Conversation) -> int:
        """Find the index of the last assistant message."""
        for i in range(len(conversation.messages) - 1, -1, -1):
            if conversation.messages[i].role == "assistant":
                return i
        return -1

    def _calculate_context_length(
        self, conversation: Conversation, last_assistant_idx: int
    ) -> int:
        """Calculate the token length of the context before the response."""
        context_messages = conversation.messages[:last_assistant_idx]
        context_parts = [msg.format() for msg in context_messages if msg.is_valid()]
        context_parts.append("<|im_start|>assistant\n")

        context_text = "\n".join(context_parts)
        context_tokens = self.tokenizer(
            context_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        return len(context_tokens["input_ids"])

    def _create_training_labels(
        self, tokenized: Dict[str, List[int]], context_length: int
    ) -> List[int]:
        """Create labels that only supervise tokens after the context."""
        labels = [-100] * len(tokenized["input_ids"])
        for i in range(context_length, len(tokenized["input_ids"])):
            labels[i] = tokenized["input_ids"][i]
        return labels

    def fits_in_context(self, text: str) -> bool:
        """Check if text fits within the maximum context length."""
        tokens = self.tokenizer(text, return_tensors=None, add_special_tokens=False)
        return len(tokens["input_ids"]) <= self.max_length - self.buffer_size


class ConversationProcessor:
    """Processes conversations into training chunks."""

    def __init__(self, tokenizer, max_length: int):
        self.chunk_builder = ChunkBuilder(tokenizer, max_length)
        self.context_overlap = 2  # Messages to keep for context between chunks

    def process_conversation(
        self, raw_messages: List[Dict[str, str]]
    ) -> List[Dict[str, List[int]]]:
        """Process a single conversation into training chunks."""
        conversation = Conversation(raw_messages)

        if not conversation.is_valid():
            return []

        if self._fits_as_single_chunk(conversation):
            return self._create_single_chunk(conversation)

        return self._create_multiple_chunks(conversation)

    def _fits_as_single_chunk(self, conversation: Conversation) -> bool:
        """Check if entire conversation fits in a single chunk."""
        full_text = conversation.format_as_text()
        return self.chunk_builder.fits_in_context(full_text)

    def _create_single_chunk(
        self, conversation: Conversation
    ) -> List[Dict[str, List[int]]]:
        """Create a single chunk from the entire conversation."""
        chunk = self.chunk_builder.build_from_conversation(
            conversation, 0, len(conversation.messages) - 1
        )
        return [chunk] if chunk else []

    def _create_multiple_chunks(
        self, conversation: Conversation
    ) -> List[Dict[str, List[int]]]:
        """Split conversation into multiple chunks at assistant responses."""
        chunks = []
        assistant_indices = conversation.get_assistant_indices()

        if not assistant_indices:
            return []

        start_idx = 0
        for end_idx in assistant_indices:
            chunk = self._find_fitting_chunk(conversation, start_idx, end_idx)
            if chunk:
                chunks.append(chunk)

            start_idx = max(start_idx, end_idx - self.context_overlap)

        return chunks

    def _find_fitting_chunk(
        self, conversation: Conversation, start_idx: int, end_idx: int
    ) -> Optional[Dict[str, List[int]]]:
        """Find the largest chunk that fits within context limits."""
        for context_start in range(start_idx, end_idx):
            test_conversation = conversation.slice(context_start, end_idx)

            if self.chunk_builder.fits_in_context(test_conversation.format_as_text()):
                return self.chunk_builder.build_from_conversation(
                    conversation, context_start, end_idx
                )

        # Fallback: try just the last exchange
        if end_idx > 0:
            return self.chunk_builder.build_from_conversation(
                conversation, end_idx - 1, end_idx
            )

        return None


class DatasetProcessor:
    """Processes entire datasets for training."""

    def __init__(
        self, conversation_processor: ConversationProcessor, batch_size: int = 100
    ):
        self.conversation_processor = conversation_processor
        self.batch_size = batch_size

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset and expand conversations into training chunks."""
        print(f"\nProcessing {len(dataset)} conversations...")

        processed_dataset = dataset.map(
            self._expand_examples,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=dataset.column_names,
        )

        # Filter empty examples
        processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)

        self._print_processing_stats(len(dataset), len(processed_dataset))
        return processed_dataset

    def _expand_examples(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Expand batch of examples into training chunks."""
        all_chunks = []

        for messages in examples["messages"]:
            chunks = self.conversation_processor.process_conversation(messages)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        return {
            "input_ids": [chunk["input_ids"] for chunk in all_chunks],
            "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
            "labels": [chunk["labels"] for chunk in all_chunks],
        }

    def _print_processing_stats(self, original_count: int, processed_count: int):
        """Print statistics about dataset processing."""
        print(
            f"Processed {original_count} conversations into {processed_count} training examples"
        )
        if processed_count > original_count:
            print(
                f"Created {processed_count - original_count} additional examples through chunking"
            )


class DatasetValidator:
    """Validates processed datasets."""

    @staticmethod
    def validate(dataset: Dataset, max_length: int):
        """Validate dataset and print information."""
        if len(dataset) == 0:
            raise ValueError(
                "No valid examples found after preprocessing. "
                "Check your data format or increase max_length."
            )

        print(f"\nDataset info:")
        print(f"- Number of examples: {len(dataset)}")
        print(f"- Features: {dataset.features}")

        if len(dataset) > 0:
            DatasetValidator._validate_sample(dataset[0], max_length)

    @staticmethod
    def _validate_sample(sample: Dict[str, List], max_length: int):
        """Validate a single sample from the dataset."""
        input_length = len(sample["input_ids"])
        supervised_tokens = sum(1 for label in sample["labels"] if label != -100)

        print(f"\nSample validation:")
        print(f"- Tokenized length: {input_length}")
        print(f"- Supervised tokens: {supervised_tokens}")

        assert len(sample["input_ids"]) == len(
            sample["labels"]
        ), "Input and label lengths mismatch"
        assert (
            input_length <= max_length
        ), f"Sample exceeds max_length: {input_length} > {max_length}"


@dataclass
class DatasetProcessingConfig:
    """Configuration that affects dataset processing."""

    jsonl_file: str
    model_name: str
    max_length: int
    dataset_batch_size: int
    
    def __post_init__(self):
        # Convert to absolute path for consistent hashing
        self.jsonl_file = str(Path(self.jsonl_file).resolve())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return asdict(self)


class DatasetCache:
    """Manages caching of processed datasets."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "dataset_cache_metadata.json"
        self.cache_file = self.cache_dir / "cached_dataset"

    def _compute_cache_key(
        self, config: DatasetProcessingConfig, dataset_hash: str
    ) -> str:
        """Compute a unique cache key based on processing configuration and dataset content."""
        config_dict = config.to_dict()
        config_dict["dataset_hash"] = dataset_hash

        # Create a stable hash of the configuration
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _compute_dataset_hash(self, dataset_path: str) -> str:
        """Compute hash of the dataset file content."""
        hasher = hashlib.sha256()
        with open(dataset_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_cached_dataset(self, config: DatasetProcessingConfig) -> Optional[Dataset]:
        """Retrieve cached dataset if it exists and is valid."""
        try:
            # Compute dataset file hash
            print(f"\nChecking cache for dataset: {config.jsonl_file}")
            dataset_hash = self._compute_dataset_hash(config.jsonl_file)
            cache_key = self._compute_cache_key(config, dataset_hash)
            print(f"Cache key: {cache_key[:16]}...")

            # Check if cache exists
            metadata = self._load_metadata()
            print(f"Found {len(metadata)} cached entries")
            
            if cache_key not in metadata:
                print("No matching cache entry found")
                return None

            cache_info = metadata[cache_key]
            cache_path = self.cache_dir / cache_info["filename"]

            if not cache_path.exists():
                print(f"Cache file {cache_path} not found, will reprocess dataset")
                return None

            # Load cached dataset
            print(f"\nLoading cached dataset from {cache_path}")
            print(f"Cache created at: {cache_info['created_at']}")
            print(f"Original dataset: {cache_info['original_file']}")

            dataset = Dataset.load_from_disk(str(cache_path))
            print(f"Loaded {len(dataset)} cached examples")

            return dataset

        except Exception as e:
            print(f"Error loading cached dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_dataset(self, dataset: Dataset, config: DatasetProcessingConfig):
        """Save processed dataset to cache."""
        try:
            # Compute cache key
            print(f"\nPreparing to cache dataset from: {config.jsonl_file}")
            dataset_hash = self._compute_dataset_hash(config.jsonl_file)
            cache_key = self._compute_cache_key(config, dataset_hash)
            print(f"Cache key: {cache_key[:16]}...")

            # Generate unique filename for this cache
            cache_filename = f"dataset_cache_{cache_key[:16]}"
            cache_path = self.cache_dir / cache_filename

            # Save dataset
            print(f"Saving processed dataset to cache: {cache_path}")
            dataset.save_to_disk(str(cache_path))

            # Update metadata
            metadata = self._load_metadata()
            metadata[cache_key] = {
                "filename": cache_filename,
                "created_at": datetime.now().isoformat(),
                "original_file": config.jsonl_file,
                "config": config.to_dict(),
                "dataset_hash": dataset_hash,
                "num_examples": len(dataset),
            }
            self._save_metadata(metadata)

            print(f"Dataset cached successfully")
            print(f"Metadata saved to: {self.metadata_file}")

        except Exception as e:
            print(f"Error saving dataset to cache: {e}")
            import traceback
            traceback.print_exc()

    def clear_cache(self):
        """Clear all cached datasets."""
        print(f"Clearing dataset cache in {self.cache_dir}")

        # Remove all cache files
        for cache_file in self.cache_dir.glob("dataset_cache_*"):
            if cache_file.is_dir():
                import shutil

                shutil.rmtree(cache_file)
            else:
                cache_file.unlink()

        # Clear metadata
        if self.metadata_file.exists():
            self.metadata_file.unlink()

        print("Cache cleared")


class PeriodicInferenceCallback(TrainerCallback):
    """Callback to run inference on a test prompt periodically during training."""
    
    def __init__(self, model_manager: ModelManager, test_prompt: str, 
                 inference_steps: int, inference_config: InferenceConfiguration):
        self.model_manager = model_manager
        self.test_prompt = test_prompt
        self.inference_steps = inference_steps
        self.inference_config = inference_config
        self.initial_response = None
    
    def on_step_end(self, args, state, control, **kwargs):
        """Run inference every N steps."""
        if state.global_step % self.inference_steps == 0 and state.global_step > 0:
            self._run_inference(state.global_step, kwargs.get("logs", {}))
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Run initial inference to establish baseline."""
        print("\n" + "="*80)
        print("Running initial inference to establish baseline...")
        print("="*80)
        self.initial_response = self._run_inference(0, {})
    
    def _run_inference(self, step: int, logs: Dict[str, float]) -> str:
        """Run inference and print results."""
        # Save the training mode
        training_mode = self.model_manager.model.training
        self.model_manager.model.eval()
        
        try:
            # Format prompt
            messages = [{"role": "user", "content": self.test_prompt}]
            formatted_prompt = self.model_manager.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate response
            with torch.no_grad():
                inputs = self.model_manager.tokenizer(
                    formatted_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048  # Use a reasonable max length for the input
                ).to(self.model_manager.model.device)
                
                outputs = self.model_manager.model.generate(
                    **inputs,
                    max_new_tokens=self.inference_config.max_new_tokens,
                    temperature=self.inference_config.temperature,
                    top_p=self.inference_config.top_p,
                    do_sample=True,
                    pad_token_id=self.model_manager.tokenizer.pad_token_id,
                    eos_token_id=self.model_manager.tokenizer.eos_token_id,
                )
                
                response = self.model_manager.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                
                # Extract only the generated part
                if formatted_prompt in response:
                    response = response[len(formatted_prompt):].strip()
            
            # Print results
            print(f"\n{'='*80}")
            print(f"Inference at step {step}")
            if logs:
                loss = logs.get('loss', 'N/A')
                print(f"Current loss: {loss}")
            print(f"{'='*80}")
            print(f"Prompt: {self.test_prompt}")
            print(f"{'-'*40}")
            print(f"Response: {response}")
            
            if step == 0:
                print(f"{'-'*40}")
                print("(This is the baseline response before training)")
            
            print(f"{'='*80}\n")
            
            return response
            
        finally:
            # Restore training mode
            if training_mode:
                self.model_manager.model.train()


class FineTuner:
    """Manages the fine-tuning process."""

    def __init__(
        self, model_manager: ModelManager, training_config: TrainingConfiguration
    ):
        self.model_manager = model_manager
        self.training_config = training_config

    def train(self, dataset: Dataset, callbacks: Optional[List[TrainerCallback]] = None):
        """Execute the training process."""
        training_args = self._create_training_arguments()

        trainer = Trainer(
            model=self.model_manager.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.model_manager.tokenizer,
            callbacks=callbacks,
        )

        trainer.train()
        self._save_results()

        # Cleanup
        del trainer
        MemoryManager.cleanup()

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from configuration."""
        return TrainingArguments(
            output_dir=self.training_config.output_dir,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            num_train_epochs=self.training_config.epochs,
            learning_rate=self.training_config.learning_rate,
            fp16=True,
            remove_unused_columns=False,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            report_to="none",
            warmup_ratio=self.training_config.warmup_ratio,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
        )

    def _save_results(self):
        """Save the trained model and adapter."""
        self.model_manager.save(self.training_config.output_dir)
        print(
            f"Training complete. Model & adapter saved to {self.training_config.output_dir}"
        )


class InferenceEngine:
    """Manages model inference."""

    def __init__(self, model_manager: ModelManager, config: InferenceConfiguration):
        self.model_manager = model_manager
        self.config = config

    def generate_response(self, prompt: str) -> str:
        """Generate a response to the given prompt."""
        formatted_prompt = self._format_prompt(prompt)

        generator = pipeline(
            "text-generation",
            model=self.model_manager.model,
            tokenizer=self.model_manager.tokenizer,
            device_map="auto",
        )

        result = generator(
            formatted_prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )

        response = self._extract_response(result[0]["generated_text"], formatted_prompt)

        # Cleanup
        del generator
        MemoryManager.cleanup()

        return response

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using chat template."""
        messages = [{"role": "user", "content": prompt}]
        return self.model_manager.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _extract_response(self, generated_text: str, prompt: str) -> str:
        """Extract the model's response from generated text."""
        return generated_text[len(prompt) :].strip()


class FineTuningApplication:
    """Main application class that orchestrates the fine-tuning process."""

    def __init__(self, args):
        self.args = args
        self.model_config = ModelConfig(name=args.model_name)
        self.model_manager = ModelManager(self.model_config)

    def run_training(self):
        """Execute the training workflow."""
        # Create dataset processing configuration
        dataset_config = DatasetProcessingConfig(
            jsonl_file=self.args.jsonl_file,
            model_name=self.args.model_name,
            max_length=self.args.max_length,
            dataset_batch_size=self.args.dataset_batch_size,
        )

        # Initialize cache
        cache = DatasetCache(self.args.output_dir)

        # Try to load cached dataset
        processed_dataset = None
        if not self.args.no_cache:
            processed_dataset = cache.get_cached_dataset(dataset_config)

        if processed_dataset is None:
            # No cache found, process dataset
            print("\nNo cached dataset found, processing from scratch...")

            # Load dataset
            dataset = self._load_dataset()

            # Setup model with LoRA
            lora_config = LoRAConfiguration(
                rank=self.args.lora_r,
                alpha=self.args.lora_alpha,
                dropout=self.args.lora_dropout,
            )

            self.model_manager.load_base_model()
            self.model_manager.apply_lora_adapter(lora_config)
            self.model_manager.prepare_for_training()

            # Process dataset
            processor = ConversationProcessor(
                self.model_manager.tokenizer, self.args.max_length
            )
            dataset_processor = DatasetProcessor(
                processor, self.args.dataset_batch_size
            )
            processed_dataset = dataset_processor.process_dataset(dataset)

            # Save to cache
            cache.save_dataset(processed_dataset, dataset_config)
        else:
            # Using cached dataset, still need to setup model
            print("\nUsing cached dataset, setting up model...")

            # Setup model with LoRA
            lora_config = LoRAConfiguration(
                rank=self.args.lora_r,
                alpha=self.args.lora_alpha,
                dropout=self.args.lora_dropout,
            )

            self.model_manager.load_base_model()
            self.model_manager.apply_lora_adapter(lora_config)
            self.model_manager.prepare_for_training()

        # Validate dataset
        DatasetValidator.validate(processed_dataset, self.args.max_length)

        # Train
        training_config = TrainingConfiguration(
            output_dir=self.args.output_dir,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            max_sequence_length=self.args.max_length,
        )

        # Setup callbacks
        callbacks = []
        if self.args.eval_prompt:
            print(f"\nPeriodic inference enabled:")
            print(f"- Evaluation prompt: {self.args.eval_prompt}")
            print(f"- Evaluation frequency: every {self.args.eval_steps} steps")
            
            inference_config = InferenceConfiguration(
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature
            )
            
            inference_callback = PeriodicInferenceCallback(
                model_manager=self.model_manager,
                test_prompt=self.args.eval_prompt,
                inference_steps=self.args.eval_steps,
                inference_config=inference_config
            )
            callbacks.append(inference_callback)

        fine_tuner = FineTuner(self.model_manager, training_config)
        fine_tuner.train(processed_dataset, callbacks=callbacks if callbacks else None)

    def run_inference(self):
        """Execute the inference workflow."""
        print(
            f"Loading model {self.model_config.name} with adapter from {self.args.output_dir}"
        )

        self.model_manager.load_base_model()
        self.model_manager.load_trained_adapter(self.args.output_dir)

        inference_config = InferenceConfiguration(
            max_new_tokens=self.args.max_new_tokens, temperature=self.args.temperature
        )

        engine = InferenceEngine(self.model_manager, inference_config)
        response = engine.generate_response(self.args.prompt)

        print("\n[Model Response]:\n", response)

    def _load_dataset(self) -> Dataset:
        """Load dataset from JSONL file."""
        try:
            dataset = load_dataset(
                "json", data_files=self.args.jsonl_file, split="train"
            )
            print(f"Loaded {len(dataset)} examples from {self.args.jsonl_file}")
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {self.args.jsonl_file}: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Fine-tuning Script",
        description="Fine-tune LLMs using LoRA with 4-bit quantization",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train", action="store_true", help="Run in training mode."
    )
    mode_group.add_argument(
        "--inference", action="store_true", help="Run in inference mode."
    )
    mode_group.add_argument(
        "--clear-cache", action="store_true", help="Clear the dataset cache."
    )

    # Model configuration
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen3-4B", help="Base model to use."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save/load model or adapter.",
    )

    # Training data
    parser.add_argument(
        "--jsonl-file",
        type=str,
        help="Path to training data (jsonl) [Required for train]",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs (train mode)."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max tokens per sample (train mode).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size per device."
    )
    parser.add_argument(
        "--dataset-batch-size",
        type=int,
        default=100,
        help="Batch size for dataset processing (different from training batch size).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force dataset reprocessing, ignoring any cached version.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate."
    )

    # LoRA parameters
    parser.add_argument(
        "--lora-r", type=int, default=16, help="LoRA attention dimension."
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha parameter."
    )
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout.")

    # Inference parameters
    parser.add_argument("--prompt", type=str, help="Prompt to run for inference mode.")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for inference."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max new tokens for inference."
    )
    
    # Training inference monitoring
    parser.add_argument(
        "--eval-prompt", 
        type=str, 
        help="Prompt to evaluate during training (optional). If provided, will run inference every N steps."
    )
    parser.add_argument(
        "--eval-steps", 
        type=int, 
        default=50,
        help="Number of training steps between evaluation prompts (default: 50)."
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Handle cache clearing
    if args.clear_cache:
        if not args.output_dir:
            raise ValueError("You must provide --output-dir to clear cache.")
        cache = DatasetCache(args.output_dir)
        cache.clear_cache()
        return

    # Validate arguments
    if args.train and not args.jsonl_file:
        raise ValueError("You must provide --jsonl-file for training mode.")
    if args.inference and not args.prompt:
        raise ValueError("You must provide --prompt for inference mode.")

    # Run application
    app = FineTuningApplication(args)

    if args.train:
        app.run_training()
    else:
        app.run_inference()


if __name__ == "__main__":
    main()
