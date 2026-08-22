import os, logging, sys
import argparse
import hashlib
from pathlib import Path

import torch
from datasets import load_from_disk
import transformers
from transformers import (
    AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--model_class", default=config.MODEL_CLASS,
                    help="Name of any model class exported by transformers "
                         "(e.g. Qwen3_5ForCausalLM, AutoModelForCausalLM, LlamaForCausalLM)")
parser.add_argument("--run_name", default=config.RUN_NAME,
                    help="W&B run name")
parser.add_argument("--fsdp_layer_cls", default=config.FSDP_LAYER_CLS,
                    help="Comma-separated decoder layer class name(s) for FSDP "
                         "transformer_layer_cls_to_wrap (must match --model_class's architecture)")
parser.add_argument("--no_activation_checkpointing", action="store_true",
                    help="Disable FSDP activation checkpointing")
args = parser.parse_args()

if not hasattr(transformers, args.model_class):
    parser.error(f"transformers has no class named {args.model_class!r}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH     = str(config.BASE_MODEL_PATH)
DATA_PATH      = str(config.TRAIN_COHORT_PATH)
VAL_DATA_PATH  = str(config.VAL_COHORT_PATH)
OUTPUT_DIR     = str(config.TRAINING_OUTPUT_DIR)
CACHE_DIR      = str(config.TOKENIZED_CACHE_DIR)
MAX_SEQ_LENGTH = 8192
WANDB_ENTITY   = config.WANDB_ENTITY
WANDB_PROJECT  = config.WANDB_PROJECT
SEED           = 42

set_seed(SEED)

def get_cache_path(data_path, model_path, max_seq_length, smoke_test):
    """Cache key depends on data file, tokenizer, seq length, and smoke_test flag."""
    data_mtime = os.path.getmtime(data_path)
    key = f"{data_path}:{data_mtime}:{model_path}:{max_seq_length}:{smoke_test}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    suffix = "_smoke" if smoke_test else ""
    return os.path.join(CACHE_DIR, f"tokenized_{h}{suffix}")

def main():
    os.environ["WANDB_ENTITY"]  = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    cache_path = get_cache_path(DATA_PATH, MODEL_PATH, MAX_SEQ_LENGTH, args.smoke_test)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No tokenized cache found at {cache_path}.\n"
            f"Run `python training/pretokenize.py"
            f"{' --smoke_test' if args.smoke_test else ''}` first."
        )
    logger.info(f"Loading cached tokenized dataset from {cache_path}")
    train_dataset = load_from_disk(cache_path)
    logger.info(f"Dataset size: {len(train_dataset):,} examples")

    val_cache_path = get_cache_path(VAL_DATA_PATH, MODEL_PATH, MAX_SEQ_LENGTH, args.smoke_test)
    if not os.path.exists(val_cache_path):
        raise FileNotFoundError(
            f"No tokenized val cache found at {val_cache_path}.\n"
            f"Run `python training/pretokenize.py"
            f"{' --smoke_test' if args.smoke_test else ''}` first."
        )
    logger.info(f"Loading cached tokenized val dataset from {val_cache_path}")
    val_dataset = load_from_disk(val_cache_path)
    logger.info(f"Val dataset size: {len(val_dataset):,} examples")

    model_class = getattr(transformers, args.model_class)
    logger.info(f"Loading model (text-only {args.model_class})...")
    model = model_class.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 3,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        learning_rate               = 2e-5,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        weight_decay                = 0.01,
        bf16                        = True,
        tf32                        = True,
        max_grad_norm               = 1.0,
        logging_steps               = 10,
        eval_strategy               = "epoch",
        per_device_eval_batch_size  = 4,
        save_strategy               = "epoch",
        save_total_limit            = None,
        optim                       = "adamw_torch_fused",
        report_to                   = "wandb",
        run_name                    = args.run_name,
        dataloader_num_workers      = 8,
        dataloader_pin_memory       = True,
        seed                        = SEED,
        fsdp        = "full_shard auto_wrap",
        fsdp_config = {
            "transformer_layer_cls_to_wrap": [c.strip() for c in args.fsdp_layer_cls.split(",")],
            "activation_checkpointing": not args.no_activation_checkpointing,
        },
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving final model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
