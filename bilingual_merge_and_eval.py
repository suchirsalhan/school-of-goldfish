#!/usr/bin/env python
"""
Unified bilingual merge + training + evaluation script.
...
"""

import argparse
import csv
import math
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import os
from huggingface_hub import upload_folder, create_repo, HfApi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# -------------------------------------------------
# HARD-CODED OPUS LANGUAGE PAIRS
# -------------------------------------------------
ALLOWED_OPUS_PAIRS = {
    "en-nl": ("en", "nl"),
    "en-es": ("en", "es"),
    "en-pl": ("en", "pl"),
    "el-en": ("el", "en"),
}

def load_opus_pair(dataset, lang_pair, split):
    if lang_pair not in ALLOWED_OPUS_PAIRS:
        raise ValueError(
            f"Unsupported lang_pair={lang_pair}. "
            f"Allowed: {list(ALLOWED_OPUS_PAIRS.keys())}"
        )
    return load_dataset(dataset, lang_pair, split=split)

# -------------------------------------------------
# DIAGNOSTIC PROMPTS
# -------------------------------------------------
FRAGMENTATION_WORDS = {
    "en": ["running", "unbelievable", "happiness", "misunderstanding", "developmental"],
    "nl": ["ontwikkeling", "onbegrijpelijk", "samenwerking", "verantwoordelijkheid"],
    "es": ["corriendo", "incomprensible", "felicidad", "responsabilidad"],
    "el": ["τρέχοντας", "ευτυχία", "ακατανόητος", "υπευθυνότητα"],
    "pl": ["szczęście", "odpowiedzialność", "niezrozumiały", "rozwojowy"],
}

ENTROPY_PROMPTS = {
    "en": "This is a test.",
    "nl": "Dit is een test.",
    "es": "Esto es una prueba.",
    "el": "Αυτό είναι ένα τεστ.",
    "pl": "To jest test.",
}

# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Bilingual merge + evaluation")
    p.add_argument("--l1", required=True)
    p.add_argument("--l2", required=True)
    p.add_argument("--model_l1", type=str)
    p.add_argument("--model_l2", type=str)
    p.add_argument("--bilingual_tokenizer", type=str)
    p.add_argument("--bgpt_model", type=str)
    p.add_argument("--dataset", default="opus_books")
    p.add_argument("--lang_pair", type=str, required=True)
    p.add_argument("--max_train_examples", type=int, default=50_000)
    p.add_argument("--max_eval_examples", type=int, default=2_000)
    p.add_argument("--do_merge", action="store_true")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--output_dir", type=str, default="out")
    p.add_argument("--push_hf", action="store_true", default=True, help="Push final checkpoint to HF hub")
    p.add_argument("--hf_repo_merged", type=str, help="HF repo ID for merged checkpoint")
    p.add_argument("--hf_repo_trained", type=str, help="HF repo ID for trained checkpoint")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--freeze_ratio", type=float, default=0.6)
    p.add_argument("--start_weight", type=float, default=0.7, help="Start weight for L1 layers in alpha(l,L)")
    p.add_argument("--end_weight", type=float, default=0.3, help="End weight for L1 layers in alpha(l,L)")
    return p.parse_args()

# -------------------------------------------------
# HF PUSH UTILITY
# -------------------------------------------------
def push_to_hf(local_folder, repo_suffix):
    username = "suchirsalhan"
    repo_name = repo_suffix
    hf_repo_id = f"{username}/{repo_name}"

    api = HfApi()
    try:
        api.repo_info(hf_repo_id)
    except Exception:
        print(f"Repo {hf_repo_id} not found. Creating it...")
        create_repo(repo_name, token=os.environ["HF_TOKEN"], repo_type="model", exist_ok=True)

    upload_folder(
        local_folder,
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message=f"Push {os.path.basename(local_folder)} checkpoint → main"
    )
    print(f"Pushed {local_folder} → HF repo {hf_repo_id}")

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def ppl(model, ds):
    model.eval()
    losses = []
    for b in ds:
        ids = torch.tensor(b["input_ids"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            losses.append(model(ids, labels=ids).loss.item())
    return math.exp(sum(losses) / len(losses))

def fragmentation(tok, lang):
    return sum(len(tok.tokenize(w)) for w in FRAGMENTATION_WORDS[lang]) / len(FRAGMENTATION_WORDS[lang])

def entropy(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[:, -1]
    p = torch.softmax(logits, dim=-1)
    return -(p * p.log()).sum().item()

def token_overlap(tok, tok_l1, tok_l2):
    v = set(tok.get_vocab())
    v1 = set(tok_l1.get_vocab())
    v2 = set(tok_l2.get_vocab())
    return len(v & v1 & v2), len(v & v1 - v2), len(v & v2 - v1)

# -------------------------------------------------
# GOLD-FISH STYLE MERGE LOGIC
# -------------------------------------------------
def merge_models(args):
    # (all your merge code remains unchanged up to saving merged checkpoint)

    # Save full merged weights immediately
    os.makedirs(args.output_dir, exist_ok=True)
    full_ckpt_dir = os.path.join(args.output_dir, "full_merged")
    model.save_pretrained(full_ckpt_dir)
    tok_new.save_pretrained(full_ckpt_dir)
    print(f"Full merged weights saved to {full_ckpt_dir}")

    # Push full merged to HF using explicit repo if provided
    if args.push_hf:
        repo_suffix = args.hf_repo_merged or f"{args.l1}_{args.l2}_merge-merged"
        push_to_hf(full_ckpt_dir, repo_suffix)

    return model, tok_new, tok_l1, tok_l2

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    args = parse_args()
    rows = []

    if args.do_merge:
        print("Merging models...")
        model, tok, tok_l1, tok_l2 = merge_models(args)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.bgpt_model).to(DEVICE)
        tok = AutoTokenizer.from_pretrained(args.bgpt_model)
        tok_l1 = AutoTokenizer.from_pretrained(args.model_l1 or args.bgpt_model)
        tok_l2 = AutoTokenizer.from_pretrained(args.model_l2 or args.bgpt_model)

    if args.do_train:
        print("Loading training data...")
        ds = load_opus_pair(args.dataset, args.lang_pair, split="train")
        n = min(args.max_train_examples, len(ds))
        ds = ds.shuffle(seed=0).select(range(n))
        ds = ds.map(lambda x: {"text": [x["translation"][args.l1], x["translation"][args.l2]]}, remove_columns=ds.column_names)
        ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=64), batched=True, remove_columns=["text"])

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=args.output_dir,
                learning_rate=args.lr,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_bs,
                gradient_accumulation_steps=args.grad_accum,
            ),
            train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        )
        trainer.train()

        # Save full trained checkpoint
        full_ckpt_dir = os.path.join(args.output_dir, "full_trained")
        model.save_pretrained(full_ckpt_dir)
        tok.save_pretrained(full_ckpt_dir)
        print(f"Full trained weights saved to {full_ckpt_dir}")

        # Push full trained to HF using explicit repo if provided
        if args.push_hf:
            repo_suffix = args.hf_repo_trained or f"{args.l1}_{args.l2}_merge-trained"
            push_to_hf(full_ckpt_dir, repo_suffix)

    if args.do_eval:
        print("Evaluating model...")
        ds = load_opus_pair(args.dataset, args.lang_pair, split="train")
        n = min(args.max_eval_examples, len(ds))
        ds = ds.select(range(n))
        ds = ds.map(lambda x: {"text": x["translation"][args.l1]}, remove_columns=ds.column_names)
        ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=64), batched=True, remove_columns=["text"])

        row = {
            "model": args.bgpt_model or args.output_dir,
            "l1": args.l1,
            "l2": args.l2,
            "ppl": ppl(model, ds),
            "fragmentation_l1": fragmentation(tok, args.l1),
            "fragmentation_l2": fragmentation(tok, args.l2),
            "entropy_l1": entropy(model, tok, ENTROPY_PROMPTS[args.l1]),
            "entropy_l2": entropy(model, tok, ENTROPY_PROMPTS[args.l2]),
        }

        s, l1o, l2o = token_overlap(tok, tok_l1, tok_l2)
        row.update({"shared_tokens": s, "l1_only_tokens": l1o, "l2_only_tokens": l2o})
        rows.append(row)

        with open("bilingual_diagnostics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(rows)

        print("Saved → bilingual_diagnostics.csv")

if __name__ == "__main__":
    main()


