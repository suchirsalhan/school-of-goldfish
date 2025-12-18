#!/usr/bin/env python3
import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Utilities: model + logprob
# -----------------------------

@dataclass
class LM:
    model_name: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device

def load_lm(model_name: str, device: str = "auto", dtype: str = "auto") -> LM:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        # common for LLaMA-style; safe default for evaluation
        tok.pad_token = tok.eos_token

    torch_dtype = None
    if dtype == "auto":
        torch_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if dev.type == "cuda" else None,
    )
    if dev.type != "cuda":
        model.to(dev)

    model.eval()
    return LM(model_name=model_name, tokenizer=tok, model=model, device=dev)

@torch.no_grad()
def loglikelihood_of_continuation(lm: LM, prompt: str, continuation: str) -> float:
    """
    Returns sum log p(continuation | prompt) over continuation tokens.
    """
    tok = lm.tokenizer

    # Encode separately to know which tokens belong to continuation
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    full_ids = tok(prompt + continuation, add_special_tokens=False).input_ids

    if len(full_ids) <= len(prompt_ids):
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=lm.device)
    outputs = lm.model(input_ids=input_ids)
    logits = outputs.logits  # [1, seq, vocab]

    # We want logprobs of tokens in positions corresponding to continuation tokens.
    # Token t at position i is predicted by logits at position i-1.
    cont_start = len(prompt_ids)
    logp = 0.0
    for pos in range(cont_start, len(full_ids)):
        token_id = full_ids[pos]
        prev_logits = logits[0, pos - 1]
        log_probs = torch.log_softmax(prev_logits, dim=-1)
        logp += float(log_probs[token_id].item())
    return logp

def pick_best_choice(lm: LM, prompt: str, choices: List[str]) -> int:
    scores = [loglikelihood_of_continuation(lm, prompt, c) for c in choices]
    return int(max(range(len(scores)), key=lambda i: scores[i]))

# -----------------------------
# Language inference from model id
# -----------------------------

ISO1_TO_ISO3 = {
    "en": "eng",
    "pl": "pol",
    "el": "ell",
    "es": "spa",
    "nl": "nld",
}

def infer_langs_from_model_name(model_name: str) -> List[str]:
    """
    Attempts to infer language pair from strings like '.../en_pl_merge-...'
    Returns ISO-639-1 codes, e.g. ['en','pl'].
    """
    m = re.search(r"/([a-z]{2})_([a-z]{2})_merge", model_name)
    if not m:
        return ["en"]
    return [m.group(1), m.group(2)]

# -----------------------------
# Benchmarks
# -----------------------------

def eval_multiblimp(lm: LM, iso3: str, max_samples: Optional[int] = None) -> Optional[float]:
    """
    MultiBLiMP minimal pairs: accuracy = P(good) > P(bad)
    Expects dataset entries to include two sentences in some fields.
    This loader is somewhat variable, so we search for likely column names.
    """
    try:
        ds = load_dataset("jumelet/multiblimp", iso3, split="train")
    except Exception:
        # Some configs may not expose as HF configs; try loading "default" and filtering.
        try:
            ds = load_dataset("jumelet/multiblimp", split="train")
            # If this works, user likely needs custom handling; return None to signal.
        except Exception:
            return None

    cols = set(ds.column_names)
    # common-ish names for minimal pair datasets
    good_keys = [k for k in ["sentence_good", "good", "grammatical", "pos", "acceptable", "correct", "sentence1", "sen"] if k in cols]
    bad_keys  = [k for k in ["sentence_bad", "bad", "ungrammatical", "neg", "unacceptable", "incorrect", "sentence2", "wrong_sen"] if k in cols]
    if not good_keys or not bad_keys:
        # last resort: try any two string columns
        str_cols = [c for c in ds.column_names if ds.features[c].dtype == "string"]
        if len(str_cols) < 2:
            return None
        good_col, bad_col = str_cols[0], str_cols[1]
    else:
        good_col, bad_col = good_keys[0], bad_keys[0]

    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    correct = 0
    for ex in tqdm(ds.select(range(n)), desc=f"MultiBLiMP({iso3})", leave=False):
        good = ex[good_col]
        bad = ex[bad_col]
        # compare unconditional sentence likelihoods (empty prompt)
        s_good = loglikelihood_of_continuation(lm, "", good)
        s_bad = loglikelihood_of_continuation(lm, "", bad)
        correct += int(s_good > s_bad)
    return correct / n if n > 0 else None

def eval_nli(lm: LM, dataset_name: str, subset: Optional[str], split: str, max_samples: Optional[int] = None) -> Optional[float]:
    """
    Zero-shot NLI via label-token likelihoods.
    Prompt: Premise ... Hypothesis ... Answer: <label>
    Labels: entailment / neutral / contradiction
    """
    try:
        ds = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
    except Exception:
        return None

    # column names differ: MNLI uses premise/hypothesis; SNLI uses premise/hypothesis.
    prem_key = "premise" if "premise"  else None
    hyp_key = "hypothesis" if "hypothesis" else None

    if prem_key is None or hyp_key is None:
        return None

    # label mapping differs; try to read ClassLabel names
    label_key = "label"
    if label_key not in ds.column_names:
        return None

    label_names = None
    try:
        feat = ds.features[label_key]
        if hasattr(feat, "names"):
            label_names = feat.names
    except Exception:
        pass

    # standard target strings
    choices = [" entailment", " neutral", " contradiction"]

    def gold_to_choice(label_val) -> Optional[int]:
        # If we know names, map by string contains
        if label_names:
            name = label_names[int(label_val)].lower()
            if "entail" in name:
                return 0
            if "neutral" in name:
                return 1
            if "contrad" in name:
                return 2
        # fallback (common for SNLI: 0 entailment, 1 neutral, 2 contradiction)
        if int(label_val) in [0, 1, 2]:
            return int(label_val)
        return None

    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    correct = 0
    total = 0
    for ex in tqdm(ds.select(range(n)), desc=f"{dataset_name}{('/'+subset) if subset else ''}", leave=False):
        gold = gold_to_choice(ex[label_key])
        if gold is None:
            continue
        prompt = (
            f"Premise: {ex[prem_key]}\n"
            f"Hypothesis: {ex[hyp_key]}\n"
            f"Answer:"
        )
        pred = pick_best_choice(lm, prompt, choices)
        correct += int(pred == gold)
        total += 1

    return (correct / total) if total > 0 else None


def eval_mubench_mc(lm: LM, dataset_name: str, subset: Optional[str], split: str, max_samples: Optional[int] = None) -> Optional[float]:
    """
    Handle MuBench multiple-choice style NLI where examples are stored as
    {'prompt': ..., 'choices': [...], 'label': ...}.
    We score choices by continuation likelihood and return accuracy.
    """
    try:
        ds = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
    except Exception:
        return None

    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    correct = 0
    total = 0
    for ex in tqdm(ds.select(range(n)), desc=f"MuBenchMC({(subset or dataset_name)})", leave=False):
        # Expect prompt, choices, label
        if not isinstance(ex, dict):
            continue
        prompt_text = ex.get("prompt", None)
        choices = ex.get("choices", None)
        label = ex.get("label", None)
        if prompt_text is None or choices is None or label is None:
            continue

        # Normalize choices to list of strings
        if isinstance(choices, dict) and "text" in choices:
            choice_list = choices["text"]
        else:
            choice_list = list(choices)

        if not choice_list:
            continue

        # Determine gold index: could be int index or a string matching a choice
        gold_idx = None
        if isinstance(label, int):
            gold_idx = int(label)
        else:
            # try to match string to one of the choices (exact or case-insensitive)
            try:
                gold_idx = choice_list.index(label)
            except Exception:
                # try lower-case matching
                lab = str(label).lower()
                for i, c in enumerate(choice_list):
                    if str(c).lower() == lab:
                        gold_idx = i
                        break

        if gold_idx is None or gold_idx < 0 or gold_idx >= len(choice_list):
            continue

        prompt = f"{prompt_text}\nAnswer:"
        # prepend space to choices to avoid tokenization merging issues
        formatted_choices = [" " + str(c) for c in choice_list]
        pred = pick_best_choice(lm, prompt, formatted_choices)
        correct += int(pred == gold_idx)
        total += 1

    return (correct / total) if total > 0 else None

def eval_arc_easy(lm: LM, iso1: Optional[str] = None, max_samples: Optional[int] = None) -> Optional[float]:
    # Try MuBench ARCEasyDataset_lighteval_{lang} first when language provided
    ds = None
    if iso1:
        cfg = f"ARCEasyDataset_lighteval_{iso1}"
        try:
            ds = load_dataset("aialt/MuBench", cfg, split="test")
        except Exception:
            ds = None

    # fallback to ai2_arc ARC-Easy
    if ds is None:
        try:
            ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
        except Exception:
            return None

    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    correct = 0
    total = 0
    for ex in tqdm(ds.select(range(n)), desc="ARC-Easy", leave=False):
        question = ex.get("prompt", "")

        # normalize choices structure
        raw_choices = ex.get("choices", None)
        choices_list = None
        letters = None
        if isinstance(raw_choices, dict):
            if "text" in raw_choices:
                choices_list = raw_choices["text"]
            elif "choices" in raw_choices and isinstance(raw_choices["choices"], list):
                choices_list = [c.get("text", c) if isinstance(c, dict) else c for c in raw_choices["choices"]]
            letters = raw_choices.get("label", None)
        elif isinstance(raw_choices, list):
            if raw_choices and isinstance(raw_choices[0], dict) and "text" in raw_choices[0]:
                choices_list = [c.get("text", "") for c in raw_choices]
                letters = [c.get("label", None) for c in raw_choices]
            else:
                choices_list = raw_choices
        else:
            # sometimes dataset uses top-level fields like 'choices_text'
            choices_list = ex.get("choices_text", None)

        if not choices_list:
            continue

        # find gold index from common keys
        answer_key = ex.get("answerKey", None)
        if answer_key is None:
            # try alternative keys
            answer_key = ex.get("answer", ex.get("label", None))

        gold = None
        if letters and answer_key is not None:
            try:
                gold = int(letters.index(answer_key))
            except Exception:
                gold = None

        if gold is None and isinstance(answer_key, int):
            gold = int(answer_key)

        if gold is None and answer_key is not None and choices_list:
            # maybe answerKey is the text of the correct answer
            try:
                gold = int(choices_list.index(answer_key))
            except Exception:
                gold = None

        if gold is None:
            continue

        prompt = f"Question: {question}\nAnswer:"
        pred = pick_best_choice(lm, prompt, [" " + str(c) for c in choices_list])
        correct += int(pred == gold)
        total += 1

    return (correct / total) if total > 0 else None

def eval_truthfulqa_mc(lm: LM, iso1: Optional[str] = None, max_samples: Optional[int] = None) -> Optional[float]:
    """
    TruthfulQA multiple-choice: choose best option by log-likelihood.
    We report MC1-style accuracy: did we pick a correct choice?
    """
    ds = None
    # try MuBench per-language config if language provided
    if iso1:
        cfg = f"TruthfulQADataset_lighteval_{iso1}"
        try:
            ds = load_dataset("aialt/MuBench", cfg, split="test")
        except Exception:
            ds = None

    # fallback to HF truthful_qa multiple_choice
    if ds is None:
        try:
            ds = load_dataset("truthful_qa", "multiple_choice", split="test")
        except Exception:
            return None

    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    correct = 0
    total = 0
    for ex in tqdm(ds.select(range(n)), desc="TruthfulQA(MC)", leave=False):
        # Support multiple dataset shapes:
        # 1) MuBench / custom: {'prompt':..., 'choices':[...], 'label': int}
        # 2) HF truthful_qa multiple_choice: 'question', 'correct_answers', 'incorrect_answers'
        # 3) HF-like with 'question', 'choices' and 'label' index
        if isinstance(ex, dict) and 'prompt' in ex and 'choices' in ex and 'label' in ex:
            prompt_text = ex.get('prompt', '')
            choices = ex.get('choices', [])
            gold = ex.get('label', None)
            if not choices or gold is None:
                continue
            prompt = f"Question: {prompt_text}\nAnswer:"
            pred_idx = pick_best_choice(lm, prompt, [" " + str(a) for a in choices])
            correct += int(pred_idx == int(gold))
            total += 1
            continue

        q = ex.get("question", "")
        # HF truthful_qa multiple_choice typically has correct_answers / incorrect_answers
        correct_answers = ex.get("correct_answers", None)
        incorrect_answers = ex.get("incorrect_answers", None)
        # HF-style alternative: 'choices' list + integer 'label'
        choices = ex.get("choices", None)
        label_idx = ex.get("label", None)

        if correct_answers is not None and incorrect_answers is not None:
            all_answers = correct_answers + incorrect_answers
            if not all_answers:
                continue
            prompt = f"Question: {q}\nAnswer:"
            pred_idx = pick_best_choice(lm, prompt, [" " + a for a in all_answers])
            picked = all_answers[pred_idx]
            correct += int(picked in set(correct_answers))
            total += 1
            continue

        if choices is not None and label_idx is not None:
            # choices may be dict with 'text' etc. handle common formats
            if isinstance(choices, dict) and 'text' in choices:
                choice_list = choices['text']
            else:
                choice_list = choices
            if not choice_list:
                continue
            prompt = f"Question: {q}\nAnswer:"
            pred_idx = pick_best_choice(lm, prompt, [" " + str(a) for a in choice_list])
            correct += int(pred_idx == int(label_idx))
            total += 1
            continue

        # Unknown format -> skip
        continue

    return (correct / total) if total > 0 else None

def eval_xcomps(lm: LM, lang_iso1: Optional[str] = None, max_samples: Optional[int] = None) -> Optional[float]:
    """
    XCOMPS is a conceptual minimal pair dataset.
    We treat it like minimal pairs: pick the more plausible sentence.
    Since column names can vary, we search for likely fields.
    """
    ds = None
    # If language provided, try language-specific split like 'comps_nl'
    if lang_iso1:
        try:
            ds = load_dataset("fpadovani/xcomps-dataset", split=f"comps_{lang_iso1}")
        except Exception:
            ds = None

    if ds is None:
        try:
            ds = load_dataset("fpadovani/xcomps-dataset", split="test")
        except Exception:
            # some datasets only provide 'validation' or 'train'
            try:
                ds = load_dataset("fpadovani/xcomps-dataset", split="validation")
            except Exception:
                return None

    cols = set(ds.column_names)
    # If dataset provides explicit acceptable/unacceptable fields, use them directly
    if "acceptable_sent" in cols and "unacceptable_sent" in cols:
        acc_col = "acceptable_sent"
        unacc_col = "unacceptable_sent"
        n = len(ds) if max_samples is None else min(max_samples, len(ds))
        correct = 0
        for ex in tqdm(ds.select(range(n)), desc=f"XCOMPS({lang_iso1})", leave=False):
            good = ex[acc_col]
            bad = ex[unacc_col]
            p_good = loglikelihood_of_continuation(lm, "", good)
            p_bad = loglikelihood_of_continuation(lm, "", bad)
            correct += int(p_good > p_bad)
        return (correct / n) if n > 0 else None

    # typical minimal-pair column guesses
    a_keys = [k for k in ["sentence1", "sent1", "a", "positive", "pos", "good"] if k in cols]
    b_keys = [k for k in ["sentence2", "sent2", "b", "negative", "neg", "bad"] if k in cols]
    label_key = "label" if "label" in cols else None

    if a_keys and b_keys and label_key:
        s1_col, s2_col = a_keys[0], b_keys[0]
        n = len(ds) if max_samples is None else min(max_samples, len(ds))
        correct = 0
        total = 0
        for ex in tqdm(ds.select(range(n)), desc=f"XCOMPS({lang_iso1})", leave=False):
            s1 = ex[s1_col]
            s2 = ex[s2_col]
            gold = int(ex[label_key])  # assume 0/1 meaning which one is correct
            # interpret: gold==0 means s1 correct, gold==1 means s2 correct (common)
            p1 = loglikelihood_of_continuation(lm, "", s1)
            p2 = loglikelihood_of_continuation(lm, "", s2)
            pred = 0 if p1 > p2 else 1
            correct += int(pred == gold)
            total += 1
        return (correct / total) if total > 0 else None

    # fallback: if no recognized protocol, return None
    return None

def eval_sib200(lm: LM, lang_code_guess: str, max_samples: Optional[int] = None) -> Optional[float]:
    """
    SIB-200 topic classification. We do generative label scoring:
      prompt = "Text: ...\nTopic:"
      choices are the label names observed in the dataset.
    We try to find the best matching config name from available configs.
    """
    try:
        configs = get_dataset_config_names("Davlan/sib200")
    except Exception:
        return None

    # Try exact match first, then substring.
    cfg = None
    if lang_code_guess in configs:
        cfg = lang_code_guess
    else:
        # common: eng_Latn, spa_Latn, ell_Grek, pol_Latn, nld_Latn, etc.
        cand = [c for c in configs if c.lower().startswith(lang_code_guess.lower())]
        if cand:
            cfg = cand[0]
        else:
            # last resort: if user passes full config elsewhere
            return None

    try:
        ds = load_dataset("Davlan/sib200", cfg, split="test")
    except Exception:
        return None

    # find text + label columns
    text_col = None
    for k in ["text", "sentence", "content", "review", "query"]:
        if k in ds.column_names:
            text_col = k
            break
    if text_col is None:
        # pick first string column
        str_cols = [c for c in ds.column_names if ds.features[c].dtype == "string"]
        if not str_cols:
            return None
        text_col = str_cols[0]

    label_col = "label" if "label" in ds.column_names else None
    if label_col is None:
        # sometimes "topic" or "category"
        for k in ["topic", "category", "class"]:
            if k in ds.column_names:
                label_col = k
                break
    if label_col is None:
        return None

    # derive label names
    label_names = None
    try:
        feat = ds.features[label_col]
        if hasattr(feat, "names"):
            label_names = feat.names
    except Exception:
        pass

    if label_names is None:
        # build from observed values
        vals = sorted(list(set(ds[label_col])))
        label_names = [str(v) for v in vals]

    # map each example label to index in label_names
    name_to_idx = {name: i for i, name in enumerate(label_names)}

    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    correct = 0
    total = 0
    for ex in tqdm(ds.select(range(n)), desc=f"SIB200({cfg})", leave=False):
        gold_val = ex[label_col]
        if isinstance(gold_val, int):
            gold = gold_val
        else:
            gold = name_to_idx.get(str(gold_val), None)
        if gold is None or gold >= len(label_names):
            continue

        prompt = f"Text: {ex[text_col]}\nTopic:"
        pred = pick_best_choice(lm, prompt, [" " + ln for ln in label_names])
        correct += int(pred == gold)
        total += 1

    return (correct / total) if total > 0 else None

def eval_bmlama(lm: LM, lang_iso1: str, max_samples: Optional[int] = None) -> Optional[float]:
    """
    BMLAMA: score gold object string by continuation likelihood vs distractors (if provided).
    Dataset formats differ; we attempt robust handling for common columns.
    """
    # Prefer BMLAMA53 (broader language coverage), fallback to BMLAMA17.
    for ds_id in ["JRQi/BMLAMA53", "JRQi/BMLAMA17"]:
        try:
            configs = get_dataset_config_names(ds_id)
            if lang_iso1 not in configs:
                continue
            ds = load_dataset(ds_id, lang_iso1, split="test")
            break
        except Exception:
            ds = None
    if ds is None:
        return None

    cols = set(ds.column_names)
    # common columns guesses
    prompt_col = None
    for k in ["prompt", "template", "query", "masked_sentence", "sentence", "Prompt"]:
        if k in cols:
            prompt_col = k
            break
    if prompt_col is None:
        # pick first string col
        str_cols = [c for c in ds.column_names if ds.features[c].dtype == "string"]
        if not str_cols:
            return None
        prompt_col = str_cols[0]

    gold_col = None
    for k in ["obj_label", "answer", "gold", "object", "target", "label", "Ans"]:
        if k in cols:
            gold_col = k
            break
    if gold_col is None:
        return None

    # optional distractors
    distractor_col = None
    for k in ["candidates", "choices", "distractors", "objects", "answers", "Candidate Ans"]:
        if k in cols:
            distractor_col = k
            break

    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    correct = 0
    total = 0
    for ex in tqdm(ds.select(range(n)), desc=f"BMLAMA({lang_iso1})", leave=False):
        prompt = ex[prompt_col]

        gold = ex[gold_col]
        if gold is None or str(gold).strip() == "":
            continue
        gold = str(gold)

        # candidates if available; handle lists or delimiter-separated strings
        if distractor_col:
            raw = ex[distractor_col]
            cands = []
            if isinstance(raw, (list, tuple)):
                cands = [str(x) for x in raw]
            elif isinstance(raw, str):
                # split common delimiters like comma, semicolon, or pipe
                parts = re.split(r"[;,|]", raw)
                cands = [p.strip() for p in parts if p.strip()]

            if cands:
                # ensure gold included
                if gold not in cands:
                    cands = [gold] + cands
                pred = pick_best_choice(lm, prompt, [" " + c for c in cands])
                correct += int(cands[pred] == gold)
                total += 1
                continue

        # Without distractors, we can't do accuracy cleanly; skip.
        continue

    return (correct / total) if total > 0 else None


# -----------------------------
# Main loop
# -----------------------------

def safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    ys = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    return sum(ys) / len(ys) if ys else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Hugging Face model ids")
    ap.add_argument("--output", required=True, help="CSV output path")
    ap.add_argument("--device", default="auto", help="cuda|cpu|auto")
    ap.add_argument("--dtype", default="auto", help="auto|fp16|bf16|fp32")
    ap.add_argument("--max_samples", type=int, default=None, help="Max examples per benchmark per language (None for all)")
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []

    for model_name in args.models:
        print(f"\n=== Loading {model_name} ===")
        lm = load_lm(model_name, device=args.device, dtype=args.dtype)

        # infer language pair
        langs_iso1 = infer_langs_from_model_name(model_name)
        langs_iso3 = [ISO1_TO_ISO3.get(x, None) for x in langs_iso1]
        langs_iso3 = [x for x in langs_iso3 if x is not None]

        # SIB200 config guesses (Flores-ish)
        sib_guess = {
            "en": "eng_Latn",
            "pl": "pol_Latn",
            "nl": "nld_Latn",
            "el": "ell_Grek",
            "es": "spa_Latn",
        }

        # For each inferred language produce one row (model-language)
        for iso1 in langs_iso1:


            # map to iso3 when available for multiblimp
            iso3 = ISO1_TO_ISO3.get(iso1)
            multiblimp = eval_multiblimp(lm, iso3, args.max_samples) if iso3 else None

            # XNLI per-language (replaces MNLI/SNLI)
            xnli = eval_nli(lm, "xnli", iso1, "test", args.max_samples)

            sib200 = None
            if iso1 in sib_guess:
                sib200 = eval_sib200(lm, sib_guess[iso1], args.max_samples)

            bmlama = eval_bmlama(lm, iso1, args.max_samples)

            # MuBench multi-MNLI / multi-SNLI configs (aialt/MuBench)
            mnli_local = eval_mubench_mc(lm, "aialt/MuBench", f"MNLIDataset_local_template_{iso1}", "test", args.max_samples)
            mnli_en = eval_mubench_mc(lm, "aialt/MuBench", f"MNLIDataset_en_template_{iso1}", "test", args.max_samples)
            snli_en = eval_mubench_mc(lm, "aialt/MuBench", f"SNLIDataset_en_template_{iso1}", "test", args.max_samples)
            snli_local = eval_mubench_mc(lm, "aialt/MuBench", f"SNLIDataset_local_template_{iso1}", "test", args.max_samples)

            truthfulqa = eval_truthfulqa_mc(lm, iso1, args.max_samples)
            xcomps = eval_xcomps(lm, iso1, args.max_samples)
            arc_easy = eval_arc_easy(lm, iso1, args.max_samples)

            row = {
                "model": model_name,
                "language": iso1,
                "multiblimp": multiblimp,
                "xnli": xnli,
                "mnli_local": mnli_local,
                "mnli_en": mnli_en,
                "snli_en": snli_en,
                "snli_local": snli_local,
                "sib200": sib200,
                "bmlama": bmlama,
                "truthfulqa": truthfulqa,
                "xcomps": xcomps,
                "arc-easy": arc_easy,
            }
            rows.append(row)

        # free VRAM between models if needed
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nWrote: {args.output}")
    print(df)

if __name__ == "__main__":
    main()


