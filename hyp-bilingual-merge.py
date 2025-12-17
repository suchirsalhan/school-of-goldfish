#!/usr/bin/env python
"""
Unified bilingual merge + training + evaluation script.
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
from huggingface_hub import upload_folder, create_repo

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
    p.add_argument("--push_hf", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--freeze_ratio", type=float, default=0.6)
    p.add_argument("--start_weight", type=float, default=0.7)
    p.add_argument("--end_weight", type=float, default=0.3)

    # 🔥 NEW
    p.add_argument(
        "--merge_mode",
        choices=["none", "top", "full"],
        default="top",
        help="Merge strategy: none=embeddings only, top=top fraction layers, full=all layers",
    )
    p.add_argument(
        "--merge_top_frac",
        type=float,
        default=0.25,
        help="Fraction of top layers to merge (used if merge_mode=top)",
    )
    return p.parse_args()

# -------------------------------------------------
# HF PUSH UTILITY
# -------------------------------------------------
def push_to_hf(local_ckpt_dir, output_dir, repo_suffix, token=None):
    base_name = os.path.basename(output_dir)
    hf_repo_id = f"suchirsalhan/{base_name}-{repo_suffix}".strip("/")

    create_repo(hf_repo_id, token=token, exist_ok=True)

    upload_folder(
        folder_path=os.path.abspath(local_ckpt_dir),
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message=f"Checkpoint: {repo_suffix}",
        token=token,
    )

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
# MERGE LOGIC
# -------------------------------------------------
def merge_models(args):
    tok_l1 = AutoTokenizer.from_pretrained(args.model_l1)
    tok_l2 = AutoTokenizer.from_pretrained(args.model_l2)
    tok_new = AutoTokenizer.from_pretrained(args.bilingual_tokenizer)

    model_l1 = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    model_l2 = AutoModelForCausalLM.from_pretrained(args.model_l2).to(DEVICE)

    d_model = model_l1.config.hidden_size
    V_new = len(tok_new)

    shared = list(set(tok_l1.get_vocab()) & set(tok_l2.get_vocab()))
    i1 = torch.tensor([tok_l1.get_vocab()[t] for t in shared], device=DEVICE)
    i2 = torch.tensor([tok_l2.get_vocab()[t] for t in shared], device=DEVICE)

    U, _, Vt = torch.linalg.svd(
        model_l1.get_input_embeddings().weight.data[i1].T
        @ model_l2.get_input_embeddings().weight.data[i2]
    )
    R = U @ Vt

    E_new = torch.zeros((V_new, d_model), device=DEVICE)

    def static_embed(model, tok, t):
        ids = tok.encode(t, add_special_tokens=False)
        return model.get_input_embeddings().weight[ids].mean(0)

    for i in range(V_new):
        t = tok_new.convert_ids_to_tokens(i)
        e1 = static_embed(model_l1, tok_l1, t)
        e2 = R @ static_embed(model_l2, tok_l2, t)
        E_new[i] = 0.5 * (e1 + e2)

    out = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    out.get_input_embeddings().weight.data = E_new.clone()
    out.lm_head.weight.data = E_new.clone()

    def alpha(l, L, sw, ew):
        return sw * (1 - l / (L - 1)) + ew * (l / (L - 1))

    L = out.config.num_hidden_layers

    if args.merge_mode == "full":
        merge_layers = range(L)
    elif args.merge_mode == "top":
        cut = int((1 - args.merge_top_frac) * L)
        merge_layers = range(cut, L)
    else:
        merge_layers = []

    for i in merge_layers:
        a = alpha(i, L, args.start_weight, args.end_weight)
        for k in out.transformer.h[i].state_dict():
            if "ln_" in k.lower() or "layernorm" in k.lower():
                continue
            out.transformer.h[i].state_dict()[k].copy_(
                a * model_l1.transformer.h[i].state_dict()[k]
                + (1 - a) * model_l2.transformer.h[i].state_dict()[k]
            )

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj"],
    )
    model = get_peft_model(out, lora)

    cut = int(L * args.freeze_ratio)
    for n, p in model.named_parameters():
        if n.startswith("transformer.h.") and int(n.split(".")[2]) < cut:
            p.requires_grad = False

    suffix = (
        "merged-full"
        if args.merge_mode == "full"
        else f"merged-top{int(100 * args.merge_top_frac)}"
        if args.merge_mode == "top"
        else "merged-none"
    )

    save_dir = os.path.join(args.output_dir, suffix)
    model.save_pretrained(save_dir)
    tok_new.save_pretrained(save_dir)

    if args.push_hf:
        push_to_hf(save_dir, args.output_dir, suffix)

    return model, tok_new, tok_l1, tok_l2

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    args = parse_args()
    rows = []

    if args.do_merge:
        model, tok, tok_l1, tok_l2 = merge_models(args)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.bgpt_model).to(DEVICE)
        tok = AutoTokenizer.from_pretrained(args.bgpt_model)
        tok_l1 = AutoTokenizer.from_pretrained(args.model_l1 or args.bgpt_model)
        tok_l2 = AutoTokenizer.from_pretrained(args.model_l2 or args.bgpt_model)

    if args.do_eval:
        ds = load_opus_pair(args.dataset, args.lang_pair, split="train")
        ds = ds.select(range(min(args.max_eval_examples, len(ds))))
        ds = ds.map(lambda x: {"text": x["translation"][args.l1]}, remove_columns=ds.column_names)
        ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=64), batched=True)

        row = {
            "merge_mode": args.merge_mode,
            "merge_top_frac": args.merge_top_frac,
            "ppl": ppl(model, ds),
            "entropy_l1": entropy(model, tok, ENTROPY_PROMPTS[args.l1]),
            "entropy_l2": entropy(model, tok, ENTROPY_PROMPTS[args.l2]),
        }
        rows.append(row)

        with open("bilingual_diagnostics.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(rows)

if __name__ == "__main__":
    main()
