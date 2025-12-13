"""
-------------   
UPDATED GOLDFISH BILINGUAL TOKENIZER TRAINING 
-------------   
For model merging experiments, we can train bilingual tokenizers for non-BGPT pairs (i.e. all lang combinations that exclude English), which are currently not pretrained. 
No temp merged files — stream directly to SentencePiece. Reduced shuffle overhead — rely on train_extremely_large_corpus=True for internal buffer shuffling. Increase threads — allow higher num_threads based on machine cores.
Separate monolingual training — optional to merge vocabs later. Parallelization — carefully balance threads per process.
"""

import os
import sentencepiece as spm
from huggingface_hub import HfApi
from datasets import load_dataset
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count

# ============================================================
# CONFIG
# ============================================================
VOCAB_SIZE = 50000
TOKENIZERS_DIR = "tokenizers"
SPM_DIR = f"{TOKENIZERS_DIR}/spm"
HF_DIR = f"{TOKENIZERS_DIR}/hf"
REPO_ID = "goldfish-models/goldfish-data"
NUM_THREADS = min(32, cpu_count())  # max threads per process

os.makedirs(SPM_DIR, exist_ok=True)
os.makedirs(HF_DIR, exist_ok=True)

api = HfApi()

# ============================================================
# LANGUAGE CODE MAP (exclude English primaries)
# ============================================================
LANG_CODES = {
    "nl": "nld_latn",
    "es": "spa_latn",
    "el": "ell_grek",
    "pl": "pol_latn",
}

CONDITIONS = ["simultaneous", "sequential"]

# ============================================================
# GET AVAILABLE TEXT FILES
# ============================================================
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
txt_files = {f.split("/")[-1]: f for f in files if f.endswith(".txt")}

def file_for_lang(code):
    fname = code + ".txt"
    if fname not in txt_files:
        raise ValueError(f"Missing file for language: {fname}")
    return txt_files[fname]

# ============================================================
# STREAMING DATASET LINES (no temp files)
# ============================================================
def iter_ds_lines(lang_files):
    ds = load_dataset(REPO_ID, split="train", streaming=True)
    for row in ds:
        text = row.get("text") or row.get("content")
        file_path = row.get("file") or row.get("path")
        if text and file_path and any(file_path.endswith(f) for f in lang_files):
            yield text.rstrip("\n")

# ============================================================
# SPM → HF CONVERSION
# ============================================================
def convert_spm_to_hf(spm_model_path, output_dir, multiple_of=2048):
    print(f"Converting {spm_model_path} to HF tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(spm_model_path, use_fast=True)

    n_to_add = 0
    if len(tokenizer) % multiple_of != 0:
        n_to_add = multiple_of - (len(tokenizer) % multiple_of)
    
    if n_to_add > 0:
        special_tokens = [f"[XXXXX{i}]" for i in range(n_to_add)]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    print(f"Added {n_to_add} tokens. Final vocab size: {len(tokenizer)}")
    tokenizer.save_pretrained(output_dir)
    print(f"HF tokenizer saved to: {output_dir}")

# ============================================================
# TOKENIZER PIPELINE
# ============================================================
def build_tokenizer(lang_pair_condition):
    lang1, lang2, condition = lang_pair_condition
    l1_code = LANG_CODES[lang1]
    l2_code = LANG_CODES[lang2]
    tokenizer_name = f"bilingual_{lang1}_{lang2}_{condition}"
    print(f"\n--- Building tokenizer: {tokenizer_name} ---")

    f1 = file_for_lang(l1_code)
    f2 = file_for_lang(l2_code)

    # --------------------------------------------------------
    # Train SentencePiece directly from stream
    # --------------------------------------------------------
    spm_prefix = os.path.join(SPM_DIR, tokenizer_name)
    print("Training SentencePiece...")
    spm.SentencePieceTrainer.train(
        input=None,
        input_sentence_size=50_000_000,  # subword sampling for speed
        train_extremely_large_corpus=True,
        shuffle_input_sentence=False,  # reduce shuffle overhead
        num_threads=NUM_THREADS,
        model_prefix=spm_prefix,
        vocab_size=VOCAB_SIZE,
        hard_vocab_limit=False,
        input_format='text',
        input_iter=iter_ds_lines([f1, f2])
    )
    print(f"SPM model saved: {spm_prefix}.model")

    # --------------------------------------------------------
    # Convert to HF tokenizer
    # --------------------------------------------------------
    hf_path = os.path.join(HF_DIR, tokenizer_name)
    os.makedirs(hf_path, exist_ok=True)
    convert_spm_to_hf(f"{spm_prefix}.model", hf_path)

# ============================================================
# GENERATE LANGUAGE PAIRS (non-English primary)
# ============================================================
langs = list(LANG_CODES.keys())
lang_pairs = [(l1, l2, cond) for l1 in langs for l2 in langs if l1 != l2 for cond in CONDITIONS]

# ============================================================
# RUN IN PARALLEL
# ============================================================
if __name__ == "__main__":
    num_processes = min(len(lang_pairs), cpu_count() // 2)  # avoid oversubscription
    with Pool(processes=num_processes) as pool:
        pool.map(build_tokenizer, lang_pairs)

    print("\nAll non-English bilingual tokenizers built successfully.")
