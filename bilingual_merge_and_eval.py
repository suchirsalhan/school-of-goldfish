#!/usr/bin/env python
"""
Unified bilingual merge + training + evaluation script.

Supports:
- Goldfish-style bilingual stabilization (Procrustes + contextual alignment)
- Configurable embedding merge strategies
- Configurable layer mixing strategies
- LoRA adaptation + layer freezing
- Optional training
- Full bilingual diagnostics: PPL, fragmentation, entropy, token overlap
- Outputs a CSV
- Fully saved merged checkpoints with automatic HF push to main branch
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
from huggingface_hub import upload_folder

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
    p.add_argument("--push_hf", action="store_true", help="Push final checkpoint to HF hub")

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
    # load tokenizers and models
    tok_l1 = AutoTokenizer.from_pretrained(args.model_l1)
    tok_l2 = AutoTokenizer.from_pretrained(args.model_l2)
    tok_new = AutoTokenizer.from_pretrained(args.bilingual_tokenizer)

    model_l1 = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    model_l2 = AutoModelForCausalLM.from_pretrained(args.model_l2).to(DEVICE)

    d_model = model_l1.config.hidden_size
    V_new = len(tok_new)

    # Procrustes
    shared = list(set(tok_l1.get_vocab()) & set(tok_l2.get_vocab()))
    i1 = torch.tensor([tok_l1.get_vocab()[t] for t in shared], device=DEVICE)
    i2 = torch.tensor([tok_l2.get_vocab()[t] for t in shared], device=DEVICE)

    U, _, Vt = torch.linalg.svd(model_l1.get_input_embeddings().weight.data[i1].T @
                                model_l2.get_input_embeddings().weight.data[i2])
    R = U @ Vt

    # Contextual embeddings
    CONTEXT_TOPK = 2000
    N_CONTEXTS = 4
    BATCH_SIZE = 64

    def contextual(model, tokenizer, tokens):
        model.eval()
        out = []
        for i in tqdm(range(0, len(tokens), BATCH_SIZE)):
            batch = tokens[i:i+BATCH_SIZE]
            ctx = [f"This contains {t}." for t in batch for _ in range(N_CONTEXTS)]
            enc = tokenizer(ctx, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                h = model(**enc, output_hidden_states=True).hidden_states[1][:, -1, :]
            for j in range(0, len(h), N_CONTEXTS):
                out.append(h[j:j+N_CONTEXTS].mean(0))
        return torch.stack(out)

    freq_tokens = shared[:CONTEXT_TOPK]
    ctx1 = contextual(model_l1, tok_l1, freq_tokens)
    ctx2 = contextual(model_l2, tok_l2, freq_tokens) @ R.T
    ctx = 0.5 * (ctx1 + ctx2)

    # Build new embeddings
    E_new = torch.zeros((V_new, d_model), device=DEVICE)

    def static_embed(model, tok, t):
        ids = tok.encode(t, add_special_tokens=False)
        return model.get_input_embeddings().weight[ids].mean(0)

    for i in tqdm(range(V_new)):
        t = tok_new.convert_ids_to_tokens(i)
        if t in freq_tokens:
            E_new[i] = ctx[freq_tokens.index(t)]
        else:
            e1 = static_embed(model_l1, tok_l1, t)
            e2 = R @ static_embed(model_l2, tok_l2, t)
            E_new[i] = 0.5 * (e1 + e2)

    # Merge transformer layers – we don't hardcode defaults for hyperparameter sweep
    def alpha(l, L, start_weight, end_weight):
        return start_weight * (1 - l/(L-1)) + end_weight * (l/(L-1))


    out = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    out.get_input_embeddings().weight.data = E_new.clone()
    out.lm_head.weight.data = E_new.clone()

    L = out.config.num_hidden_layers
    for i in range(L):
        a = alpha(i, L, args.start_weight, args.end_weight)
        for k in out.transformer.h[i].state_dict():
            out.transformer.h[i].state_dict()[k].copy_(
                a * model_l1.transformer.h[i].state_dict()[k] +
                (1-a) * model_l2.transformer.h[i].state_dict()[k]
            )

    # LoRA + freeze
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                      target_modules=["c_attn","c_proj","mlp.c_fc","mlp.c_proj"])
    model = get_peft_model(out, lora)

    cut = int(L * args.freeze_ratio)
    for nme, p in model.named_parameters():
        if nme.startswith("transformer.h.") and int(nme.split(".")[2]) < cut:
            p.requires_grad = False
        else:
            p.requires_grad = True

    # Save full merged weights immediately
    os.makedirs(args.output_dir, exist_ok=True)
    full_ckpt_dir = os.path.join(args.output_dir, "full_merged")
    model.save_pretrained(full_ckpt_dir)
    tok_new.save_pretrained(full_ckpt_dir)
    print(f"Full merged weights saved to {full_ckpt_dir}")

    # Automatic push to HF main branch
    if args.push_hf:
        upload_folder(full_ckpt_dir, repo_id=args.output_dir.split("/")[-1], repo_type="model", commit_message="Full merged checkpoint → main")
        print(f"Pushed merged model to HF main branch")

    return model, tok_new, tok_l1, tok_l2

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    args = parse_args()
    rows = []

    # -------------------------------------------------
    # LOAD / MERGE MODEL
    # -------------------------------------------------
    if args.do_merge:
        print("Merging models...")
        model, tok, tok_l1, tok_l2 = merge_models(args)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.bgpt_model).to(DEVICE)
        tok = AutoTokenizer.from_pretrained(args.bgpt_model)
        tok_l1 = AutoTokenizer.from_pretrained(args.model_l1 or args.bgpt_model)
        tok_l2 = AutoTokenizer.from_pretrained(args.model_l2 or args.bgpt_model)

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------
    if args.do_train:
        print("Loading training data...")
        ds = load_opus_pair(args.dataset, args.lang_pair, split="train")

        n = min(args.max_train_examples, len(ds))
        print(f"Using {n}/{len(ds)} training examples")
        ds = ds.shuffle(seed=0).select(range(n))

        def flatten(x):
            return {
                "text": [
                    x["translation"][args.l1],
                    x["translation"][args.l2],
                ]
            }

        ds = ds.map(flatten, remove_columns=ds.column_names)
        ds = ds.map(
            lambda x: tok(x["text"], truncation=True, max_length=64),
            batched=True,
            remove_columns=["text"],
        )

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

        # Save full checkpoint after training
        full_ckpt_dir = os.path.join(args.output_dir, "full_trained")
        model.save_pretrained(full_ckpt_dir)
        tok.save_pretrained(full_ckpt_dir)
        print(f"Full trained weights saved to {full_ckpt_dir}")

        # Automatic push to HF main branch
        if args.push_hf:
            upload_folder(full_ckpt_dir, repo_id=args.output_dir.split("/")[-1], repo_type="model", commit_message="Full trained checkpoint → main")
            print(f"Pushed trained model to HF main branch")

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    if args.do_eval:
        print("Evaluating model...")
        ds = load_opus_pair(args.dataset, args.lang_pair, split="train")

        n = min(args.max_eval_examples, len(ds))
        ds = ds.select(range(n))

        def flatten_eval(x):
            return {"text": x["translation"][args.l1]}

        ds = ds.map(flatten_eval, remove_columns=ds.column_names)
        ds = ds.map(
            lambda x: tok(x["text"], truncation=True, max_length=64),
            batched=True,
            remove_columns=["text"],
        )

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
        row.update(
            {
                "shared_tokens": s,
                "l1_only_tokens": l1o,
                "l2_only_tokens": l2o,
            }
        )
        rows.append(row)

        with open("bilingual_diagnostics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(rows)

        print("Saved → bilingual_diagnostics.csv")

if __name__ == "__main__":
    main()


