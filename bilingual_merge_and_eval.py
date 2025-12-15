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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

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

    # Languages
    p.add_argument("--l1", required=True)
    p.add_argument("--l2", required=True)

    # Models
    p.add_argument("--model_l1", type=str)
    p.add_argument("--model_l2", type=str)
    p.add_argument("--bilingual_tokenizer", type=str)
    p.add_argument("--bgpt_model", type=str)

    # Dataset
    p.add_argument("--dataset", default="opus_books")
    p.add_argument("--lang_pair", type=str)
    p.add_argument("--max_train_examples", type=int, default=50_000)
    p.add_argument("--max_eval_examples", type=int, default=2_000)

    # Merge / training / evaluation
    p.add_argument("--do_merge", action="store_true")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--output_dir", type=str, default="out")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)

    # -------------------------------------------------
    # Merge hyperparameters
    # -------------------------------------------------
    # Embedding
    p.add_argument("--embed_merge", default="hybrid", choices=["static","contextual","hybrid"])
    p.add_argument("--embed_lambda", type=float, default=0.5)

    # Shared vocab
    p.add_argument("--shared_vocab_strategy", default="intersection", choices=["intersection","topk_freq"])
    p.add_argument("--context_topk", type=int, default=2000)

    # Contextualization
    p.add_argument("--context_template", default="simple", choices=["simple","definition"])
    p.add_argument("--context_layer", default="layer1", choices=["input","layer1","layerN"])
    p.add_argument("--n_contexts", type=int, default=4)

    # Layer merging
    p.add_argument("--layer_merge", default="linear", choices=["linear","flat","early","late"])
    p.add_argument("--alpha_low", type=float, default=0.7)
    p.add_argument("--alpha_high", type=float, default=0.3)

    # LoRA / adaptation
    p.add_argument("--freeze_ratio", type=float, default=0.6)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--train_embeddings", action="store_true")

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
# MERGE HELPERS
# -------------------------------------------------
def select_shared(tok1, tok2, strategy, topk):
    shared = list(set(tok1.get_vocab()) & set(tok2.get_vocab()))
    if strategy == "topk_freq":
        shared = shared[:topk]
    return shared

def make_context(token, template):
    if template == "simple":
        return f"This contains {token}."
    elif template == "definition":
        return f"The meaning of {token} is"
    return token

def contextual_embed(model, tokenizer, tokens, args):
    model.eval()
    outs = []
    layer_map = {"input":0,"layer1":1,"layerN":-1}
    layer_idx = layer_map[args.context_layer]
    for t in tokens:
        ctx = [make_context(t, args.context_template) for _ in range(args.n_contexts)]
        enc = tokenizer(ctx, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[layer_idx]
        outs.append(h[:, -1].mean(0))
    return torch.stack(outs)

def static_embed(model, tok, token):
    ids = tok.encode(token, add_special_tokens=False)
    if len(ids) == 0:
        return torch.zeros(model.config.hidden_size, device=model.device)
    return model.get_input_embeddings().weight[ids].mean(0)

def procrustes(E1, E2, i1, i2):
    U, _, Vt = torch.linalg.svd(E1[i1].T @ E2[i2])
    return U @ Vt

def merge_embedding(static1, static2, ctx1, ctx2, R, args):
    e_static = 0.5*(static1 + R @ static2)
    e_ctx = 0.5*(ctx1 + ctx2)
    if args.embed_merge=="static":
        return e_static
    if args.embed_merge=="contextual":
        return e_ctx
    return args.embed_lambda * e_ctx + (1-args.embed_lambda)*e_static

def layer_alpha(i, L, args):
    if args.layer_merge=="flat":
        return args.alpha_low
    if args.layer_merge=="early":
        return 1.0 if i<L//2 else 0.0
    if args.layer_merge=="late":
        return 0.0 if i<L//2 else 1.0
    # linear
    return args.alpha_low*(1-i/(L-1)) + args.alpha_high*(i/(L-1))

def freeze_lower_layers(model, ratio):
    n = model.config.num_hidden_layers
    cut = int(n*ratio)
    for name, p in model.named_parameters():
        p.requires_grad = True
        if name.startswith("transformer.h."):
            if int(name.split(".")[2]) < cut:
                p.requires_grad = False
    return model


# -------------------------------------------------
# MERGE MODELS
# -------------------------------------------------
def merge_models(args):
    tok1 = AutoTokenizer.from_pretrained(args.model_l1)
    tok2 = AutoTokenizer.from_pretrained(args.model_l2)
    tok_new = AutoTokenizer.from_pretrained(args.bilingual_tokenizer)

    m1 = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    m2 = AutoModelForCausalLM.from_pretrained(args.model_l2).to(DEVICE)

    shared = select_shared(tok1, tok2, args.shared_vocab_strategy, args.context_topk)
    i1 = torch.tensor([tok1.get_vocab()[t] for t in shared], device=DEVICE)
    i2 = torch.tensor([tok2.get_vocab()[t] for t in shared], device=DEVICE)

    R = procrustes(m1.get_input_embeddings().weight.data,
                   m2.get_input_embeddings().weight.data,
                   i1, i2)

    # Contextual embeddings
    ctx1 = contextual_embed(m1, tok1, shared, args)
    ctx2 = contextual_embed(m2, tok2, shared, args) @ R.T

    # Build new embedding matrix
    V_new = len(tok_new)
    d_model = m1.config.hidden_size
    E_new = torch.zeros((V_new, d_model), device=DEVICE)

    for i in tqdm(range(V_new), desc="Building embedding matrix"):
        t = tok_new.convert_ids_to_tokens(i)
        s = shared.index(t) if t in shared else None
        e_static1 = static_embed(m1, tok1, t)
        e_static2 = static_embed(m2, tok2, t)
        if s is not None:
            e_ctx1 = ctx1[s]
            e_ctx2 = ctx2[s]
        else:
            e_ctx1 = torch.zeros(d_model, device=DEVICE)
            e_ctx2 = torch.zeros(d_model, device=DEVICE)
        E_new[i] = merge_embedding(e_static1, e_static2, e_ctx1, e_ctx2, R, args)

    # Merge transformer layers
    out = AutoModelForCausalLM.from_pretrained(args.model_l1).to(DEVICE)
    out.get_input_embeddings().weight.data = E_new.clone()
    out.lm_head.weight.data = E_new.clone()

    L = out.config.num_hidden_layers
    for i in range(L):
        a = layer_alpha(i, L, args)
        for k in out.transformer.h[i].state_dict():
            out.transformer.h[i].state_dict()[k].copy_(
                a*m1.transformer.h[i].state_dict()[k] + (1-a)*m2.transformer.h[i].state_dict()[k]
            )

    # LoRA adaptation
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn","c_proj","mlp.c_fc","mlp.c_proj"]
    )
    model = get_peft_model(out, lora)
    model = freeze_lower_layers(model, args.freeze_ratio)

    return model, tok_new, tok1, tok2


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    args = parse_args()
    rows = []

    # Merge
    if args.do_merge:
        print("Merging models...")
        model, tok, tok_l1, tok_l2 = merge_models(args)
    else:
        print("Loading existing B-GPT...")
        model = AutoModelForCausalLM.from_pretrained(args.bgpt_model).to(DEVICE)
        tok = AutoTokenizer.from_pretrained(args.bgpt_model)
        tok_l1 = AutoTokenizer.from_pretrained(args.model_l1 or args.bgpt_model)
        tok_l2 = AutoTokenizer.from_pretrained(args.model_l2 or args.bgpt_model)

    # Training
    if args.do_train:
        print("Loading training data...")
        ds = load_dataset(args.dataset, args.lang_pair, split="train")
        ds = ds.shuffle(seed=0).select(range(args.max_train_examples))

        def flatten(x):
            return {"text": list(x["translation"].values())}

        ds = ds.map(flatten, remove_columns=ds.column_names).flatten_indices()
        ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=64),
                    batched=True, remove_columns=["text"])

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

    # Evaluation
    if args.do_eval:
        print("Evaluating model...")
        ds = load_dataset(args.dataset, args.lang_pair, split=f"train[:{args.max_eval_examples}]")

        def flatten(x):
            return {"text": x["translation"][args.l1]}

        ds = ds.map(flatten, remove_columns=ds.column_names)
        ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=64),
                    batched=True, remove_columns=["text"])

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

        s,l1o,l2o = token_overlap(tok, tok_l1, tok_l2)
        row.update({"shared_tokens":s,"l1_only_tokens":l1o,"l2_only_tokens":l2o})
        rows.append(row)

        with open("bilingual_diagnostics.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(rows)

        print("Saved → bilingual_diagnostics.csv")

if __name__=="__main__":
    main()
