#!/usr/bin/env python
"""
Unified bilingual tokenwise merge + training + evaluation script.
Pushes merged weights as 'TOKENWISE_MERGE' on Hugging Face hub.
Supports a parameter to enable Tokenizer-Aware Embedding Merge.
LoRA + layer freezing only active during training.
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

ALLOWED_OPUS_PAIRS = {
    "en-nl": ("en", "nl"),
    "en-es": ("en", "es"),
    "en-pl": ("en", "pl"),
    "el-en": ("el", "en"),
}

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

def parse_args():
    p = argparse.ArgumentParser("Bilingual tokenwise merge + evaluation")
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
    p.add_argument("--alpha_l1", type=float, default=0.5, help="Tokenwise interpolation weight for L1 embeddings")
    p.add_argument("--tokenizer_aware_merge", action="store_true",  default=True, help="Use Tokenizer-Aware Embedding Merge")
    return p.parse_args()

def push_to_hf(local_ckpt_dir, output_dir, l1=None, l2=None, token=None):
    # include l1 and l2 in the repo name
    base_name = os.path.basename(output_dir)
    if l1 and l2:
        hf_repo_id = f"suchirsalhan/{l1}_{l2}-{base_name}-TOKENWISE_MERGE"
    else:
        hf_repo_id = f"suchirsalhan/{base_name}-TOKENWISE_MERGE"
    try:
        create_repo(hf_repo_id, token=token, exist_ok=True)
        print(f"Repo ready: {hf_repo_id}")
    except Exception as e:
        print(f"Warning: could not create repo {hf_repo_id}: {e}")
    print(f"Pushing {local_ckpt_dir} → {hf_repo_id}")
    upload_folder(
        folder_path=os.path.abspath(local_ckpt_dir),
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Full tokenwise merged checkpoint → main",
        token=token,
    )
    print(f"Pushed TOKENWISE_MERGE model to HF: {hf_repo_id}")


def load_opus_pair(dataset, lang_pair, split):
    if lang_pair not in ALLOWED_OPUS_PAIRS:
        raise ValueError(f"Unsupported lang_pair={lang_pair}. Allowed: {list(ALLOWED_OPUS_PAIRS.keys())}")
    return load_dataset(dataset, lang_pair, split=split)

def ppl(model, ds):
    model.eval()
    losses = []
    for b in ds:
        ids = torch.tensor(b["input_ids"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            losses.append(model(ids, labels=ids).loss.item())
    return math.exp(sum(losses)/len(losses))

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

def merge_models(args):
    tok_l1 = AutoTokenizer.from_pretrained(args.model_l1)
    tok_l2 = AutoTokenizer.from_pretrained(args.model_l2)
    tok_new = AutoTokenizer.from_pretrained(args.bilingual_tokenizer)

    model_l1 = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    model_l2 = AutoModelForCausalLM.from_pretrained(args.model_l2).to(DEVICE)

    d_model = model_l1.config.hidden_size
    V_new = len(tok_new)
    E_new = torch.zeros((V_new, d_model), device=DEVICE)

    if args.tokenizer_aware_merge:
        def static_embed(model, tok, token):
            ids = tok.encode(token, add_special_tokens=False)
            if len(ids) == 0:
                return torch.zeros(d_model, device=DEVICE)
            return model.get_input_embeddings()(torch.tensor(ids).to(DEVICE)).mean(0)

        for i in tqdm(range(V_new), desc="Tokenwise embedding merge"):
            token = tok_new.convert_ids_to_tokens(i)
            e1 = static_embed(model_l1, tok_l1, token)
            e2 = static_embed(model_l2, tok_l2, token)
            E_new[i] = args.alpha_l1 * e1 + (1 - args.alpha_l1) * e2

    # Inject merged embeddings into new model
    model_merged = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    if args.tokenizer_aware_merge:
        model_merged.get_input_embeddings().weight.data = E_new.clone()
        model_merged.lm_head.weight.data = E_new.clone()

    # LoRA + freeze only during training
    if args.do_train:
        L = model_merged.config.num_hidden_layers
        lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                          target_modules=["c_attn","c_proj","mlp.c_fc","mlp.c_proj"])
        model_merged = get_peft_model(model_merged, lora)

        cut = int(L * args.freeze_ratio)
        for nme, p in model_merged.named_parameters():
            if nme.startswith("transformer.h.") and int(nme.split(".")[2]) < cut:
                p.requires_grad = False
            else:
                p.requires_grad = True

    full_merged_dir = os.path.join(args.output_dir, "full_tokenwise_merged")
    model_merged.save_pretrained(full_merged_dir)
    tok_new.save_pretrained(full_merged_dir)
    print(f"Tokenwise merged weights saved to {full_merged_dir}")
    if args.push_hf:
        push_to_hf(full_merged_dir, args.output_dir, l1=args.l1, l2=args.l2)
    return model_merged, tok_new, tok_l1, tok_l2

def main():
    args = parse_args()
    rows = []

    if args.do_merge:
        print("Merging models (tokenwise)...")
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

        full_trained_dir = os.path.join(args.output_dir, "full_trained")
        model.save_pretrained(full_trained_dir)
        tok.save_pretrained(full_trained_dir)
        print(f"Full trained weights saved to {full_trained_dir}")
        if args.push_hf:
            push_to_hf(full_merged_dir, args.output_dir, l1=args.l1, l2=args.l2)

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
