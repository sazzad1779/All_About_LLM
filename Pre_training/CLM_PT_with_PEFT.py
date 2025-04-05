#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""


"""
Fine-tuning script for causal language modeling (CLM) with support for:
- Full fine-tuning or LoRA (Parameter-Efficient Fine-Tuning)
- Quantization (4-bit/8-bit)
- Flash Attention 2
- Distributed training
- Text data processing from multiple sources
"""

import logging
import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path

import datasets
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training

# Ensure minimum transformers version
check_min_version("4.40.0")

# Configure logging
logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


#################################
# Argument Classes
#################################

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer for weights initialization."}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": f"Model type from the list: {', '.join(MODEL_TYPES)}"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name if different from model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name if different from model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store downloaded pretrained models"}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use fast tokenizer (tokenizers library)"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Use the token generated from huggingface-cli login"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override default torch.dtype for model loading",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "Create model as empty shell, load parameters when needed"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments for dataset configuration"""
    dataset_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Directory containing dataset text files"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate number of training examples for debugging"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate number of evaluation examples for debugging"}
    )
    streaming: bool = field(
        default=False, 
        metadata={"help": "Enable streaming mode"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Sequence length after tokenization"}
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite cached datasets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={"help": "Percentage of train set to use as validation"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing"}
    )
    keep_linebreaks: bool = field(
        default=True, 
        metadata={"help": "Preserve line breaks in text files"}
    )
    data_cache_dir: Optional[str] = field(
        default="./", 
        metadata={"help": "Directory to store processed datasets"}
    )


@dataclass
class PeftTrainingArguments(TrainingArguments):
    """Arguments for PEFT (Parameter-Efficient Fine-Tuning)"""
    trainable: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of modules to train in LoRA"}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    lora_alpha: Optional[float] = field(
        default=32.,
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of modules to save separately"}
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Run with minimal data for debugging"}
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained PEFT adapters"}
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Use Flash Attention 2 implementation"}
    )
    double_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Use double quantization for 4-bit quantization"}
    )
    quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization data type (fp4 or nf4)"}
    )
    load_in_kbits: Optional[int] = field(
        default=16,
        metadata={"help": "Load model in 4/8/16-bit precision"}
    )
    full_finetuning: Optional[bool] = field(
        default=False,
        metadata={"help": "Perform full fine-tuning instead of PEFT"}
    )


#################################
# Helper Functions
#################################

def compute_metrics(eval_preds):
    """Compute accuracy metrics for evaluation"""
    preds, labels = eval_preds
    # Shift labels and predictions for sequence prediction
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return {"accuracy": float(accuracy_score(labels, preds))}


def preprocess_logits_for_metrics(logits, labels):
    """Convert logits to predictions for metric computation"""
    if isinstance(logits, tuple):
        logits = logits[0]  # Get first tensor from tuple if needed
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    """Data collator with fault tolerance for corrupted examples"""
    # Convert non-Mapping features to dictionaries
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    
    first = features[0]
    batch = {}

    # Handle labels with appropriate data type
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handle all other keys
    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:
        # Fall back to using first example for corrupted batches
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


#################################
# Dataset Processing
#################################

def load_and_process_datasets(tokenizer, data_args, training_args, block_size):
    """Load and process datasets from text files"""
    # Setup logging for dataset processing
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    
    def tokenize_function(examples):
        """Tokenize text examples"""
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # Log warning about long sequences
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked"
                " before being passed to the model."
            )
        return output
    
    def group_texts(examples):
        """Concatenate and chunk text into blocks of block_size"""
        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Truncate to multiple of block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Split by chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Process all text files in dataset directory
    lm_datasets = []
    path = Path(data_args.dataset_dir)
    files = [file.name for file in path.glob("*.txt")]
    
    # Limit to first file in debug mode
    if training_args.debug_mode:
        files = [files[0]]
    
    # Process each file
    for idx, file in enumerate(files):
        data_file = os.path.join(path, file)
        filename = ''.join(file.split(".")[:-1])
        cache_path = os.path.join(data_args.data_cache_dir, filename+f"_{block_size}")
        os.makedirs(cache_path, exist_ok=True)
        
        try:
            # Try to load preprocessed dataset from disk
            processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
            logger.info(f'Training dataset-{filename} loaded from disk cache')
        except Exception:
            # Process dataset from scratch
            cache_dir = os.path.join(data_args.data_cache_dir, filename+f"_text_{block_size}")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load raw text dataset
            raw_dataset = load_dataset(
                "text", 
                data_files=data_file, 
                cache_dir=cache_dir, 
                keep_in_memory=False
            )
            logger.info(f"{file} has been loaded")
            
            # Tokenize dataset
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns="text",
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names={k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                desc="Tokenizing dataset",
            )
            
            # Group into blocks
            grouped_datasets = tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names={k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                desc=f"Grouping texts in chunks of {block_size}",
            )
            
            processed_dataset = grouped_datasets
            processed_dataset.save_to_disk(cache_path)
        
        # Combine datasets
        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
    
    # Split into train and validation
    lm_datasets = lm_datasets.train_test_split(test_size=data_args.validation_split_percentage)
    
    # Prepare train dataset
    train_dataset = None
    if training_args.do_train:
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Number of training samples: {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    
    # Prepare evaluation dataset
    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = lm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Number of evaluation samples: {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
    
    return train_dataset, eval_dataset


#################################
# Model Setup
#################################

def setup_model_and_tokenizer(model_args, training_args):
    """Configure and load the model and tokenizer"""
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You must specify a tokenizer with --tokenizer_name or --tokenizer_name_or_path."
        )
    
    # Load config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("Creating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
    
    # Setup quantization if needed
    compute_dtype = (
        torch.float16 if training_args.fp16 else 
        (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    quantization_config = None
    if training_args.load_in_kbits in [4, 8]:
        # Configure modules to skip in quantization
        load_in_8bit_skip_modules = None
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type  # 'fp4' or 'nf4'
        )
        logger.info(f"Quantization config: {quantization_config.to_dict()}")
    
    # Setup model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        
        # Set device map based on environment
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        
        # Load pretrained model
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            device_map=device_map,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if training_args.use_flash_attention_2 else "sdpa"
        )
    else:
        # Create new model from config
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    # Prepare model for k-bit training if needed
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    
    # Disable caching during training
    model.config.use_cache = False
    
    # Ensure model and tokenizer vocab sizes match
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    
    if model_vocab_size != tokenizer_vocab_size:
        logger.info(f"Resizing model vocab size to {tokenizer_vocab_size}")
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer, device_map


#################################
# PEFT Configuration
#################################

def configure_peft(model, training_args, device_map):
    """Configure Parameter-Efficient Fine-Tuning (PEFT)"""
    if not training_args.full_finetuning:
        if training_args.peft_path is not None:
            logger.info(f"Loading pre-trained PEFT adapters from {training_args.peft_path}")
            model = PeftModel.from_pretrained(
                model, 
                training_args.peft_path, 
                device_map=device_map, 
                is_trainable=True
            )
        else:
            logger.info("Initializing new PEFT adapters")
            target_modules = training_args.trainable.split(',')
            
            modules_to_save = None
            if training_args.modules_to_save is not None:
                modules_to_save = training_args.modules_to_save.split(',')
            
            logger.info(f"Target modules for LoRA: {target_modules}")
            logger.info(f"LoRA rank: {training_args.lora_rank}")
            logger.info(f"Modules to save separately: {modules_to_save}")
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=training_args.lora_rank, 
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                modules_to_save=modules_to_save
            )
            
            model = get_peft_model(model, peft_config)
        
        # Print trainable parameters info
        model.print_trainable_parameters()
    
    return model


#################################
# Main Function
#################################

def main():
    """Main training function"""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftTrainingArguments))
    
    # Handle JSON file input
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Set log levels
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log training setup
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Check for existing checkpoints
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, "
                "change the `--output_dir` or add `--overwrite_output_dir`."
            )
    
    # Set random seed
    set_seed(training_args.seed)
    
    # Setup model and tokenizer
    model, tokenizer, device_map = setup_model_and_tokenizer(model_args, training_args)
    
    # Determine block size for text chunking
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default "
                "`block_size` value of 1024. Using block_size=1024 instead."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    
    # Load and process datasets
    train_dataset, eval_dataset = load_and_process_datasets(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        block_size=block_size
    )
    
    # Apply PEFT configuration if needed
    model = configure_peft(model, training_args, device_map)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too
        
        # Log and save metrics
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None 
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate()
        
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None 
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()