"""
Pre-tokenize training data and cache to disk.
"""
import os, json, logging, sys
import argparse
import hashlib
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--force", action="store_true",
                    help="Rebuild cache even if it already exists")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH     = str(config.BASE_MODEL_PATH)
DATA_PATH      = str(config.TRAIN_COHORT_PATH)
VAL_DATA_PATH  = str(config.VAL_COHORT_PATH)
CACHE_DIR      = str(config.TOKENIZED_CACHE_DIR)
MAX_SEQ_LENGTH = 8192

def get_cache_path(data_path, model_path, max_seq_length, smoke_test):
    data_mtime = os.path.getmtime(data_path)
    key = f"{data_path}:{data_mtime}:{model_path}:{max_seq_length}:{smoke_test}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    suffix = "_smoke" if smoke_test else ""
    return os.path.join(CACHE_DIR, f"tokenized_{h}{suffix}")

def load_notes(path):
    texts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            note = row.get("Note", "").strip()
            if note:
                texts.append(note)
    logger.info(f"Loaded {len(texts):,} notes")
    return texts

def tokenize_notes(texts, tokenizer):
    logger.info("Tokenizing...")
    encoded = tokenizer(
        texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )

    full_lengths = [
        len(ids) for ids in tokenizer(
            texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )["input_ids"]
    ]
    truncated = sum(1 for l in full_lengths if l > MAX_SEQ_LENGTH)
    total     = len(texts)
    logger.info(f"Total notes:      {total:,}")
    logger.info(f"Truncated notes:  {truncated:,} ({100 * truncated / total:.1f}%)")
    logger.info(f"Mean length:      {sum(full_lengths) // total:,} tokens")
    logger.info(f"Max length:       {max(full_lengths):,} tokens")

    return Dataset.from_dict({"input_ids": encoded["input_ids"]})

def cache_split(data_path, tokenizer):
    cache_path = get_cache_path(data_path, MODEL_PATH, MAX_SEQ_LENGTH, args.smoke_test)
    logger.info(f"Cache path: {cache_path}")

    if os.path.exists(cache_path) and not args.force:
        logger.info(f"Cache already exists at {cache_path} — nothing to do.")
        logger.info("Pass --force to rebuild.")
        return

    if os.path.exists(cache_path) and args.force:
        import shutil
        logger.info(f"--force specified, removing existing cache at {cache_path}")
        shutil.rmtree(cache_path)

    texts = load_notes(data_path)
    if args.smoke_test:
        texts = texts[:10]
        logger.info("Smoke test mode: using 10 notes only")

    dataset = tokenize_notes(texts, tokenizer)

    os.makedirs(CACHE_DIR, exist_ok=True)
    dataset.save_to_disk(cache_path)
    logger.info(f"Cached tokenized dataset to {cache_path}")
    logger.info(f"Dataset size: {len(dataset):,} examples")

def main():
    logger.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("=== Train split ===")
    cache_split(DATA_PATH, tokenizer)
    logger.info("=== Val split ===")
    cache_split(VAL_DATA_PATH, tokenizer)

if __name__ == "__main__":
    main()
