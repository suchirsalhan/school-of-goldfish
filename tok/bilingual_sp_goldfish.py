# Bilingual Goldfish SentencePiece Tokenizers

import os
import sentencepiece as spm
from huggingface_hub import HfApi
from datasets import load_dataset
from transformers import AlbertTokenizer

# ============================================================
# CONFIG
# ============================================================
VOCAB_SIZE = 50000
TOKENIZERS_DIR = "tokenizers"
SPM_DIR = f"{TOKENIZERS_DIR}/spm"
HF_DIR = f"{TOKENIZERS_DIR}/hf"
REPO_ID = "goldfish-models/goldfish-data"

os.makedirs(SPM_DIR, exist_ok=True)
os.makedirs(HF_DIR, exist_ok=True)

api = HfApi()

# ============================================================
# LANGUAGE CODE MAP
# ============================================================
LANG_CODES = {
    "en": "eng_latn",
    "nl": "nld_latn",
    "es": "spa_latn",
    "el": "ell_grek",
    "pl": "pol_latn",
}

# ============================================================
# B-GPT CONFIGS
# ============================================================
BGPT_MODELS = [
    ("en", "nl", "simultaneous"),
    ("nl", "en", "simultaneous"),
    ("en", "nl", "sequential"),
    ("nl", "en", "sequential"),

    ("en", "es", "simultaneous"),
    ("es", "en", "simultaneous"),
    ("en", "es", "sequential"),
    ("es", "en", "sequential"),

    ("en", "el", "simultaneous"),
    ("el", "en", "simultaneous"),
    ("en", "el", "sequential"),
    ("el", "en", "sequential"),

    ("en", "pl", "simultaneous"),
    ("pl", "en", "simultaneous"),
    ("en", "pl", "sequential"),
    ("pl", "en", "sequential"),
]

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
# SPM → HF CONVERSION (INLINED)
# ============================================================
def convert_spm_to_hf(
    spm_model_path,
    output_dir,
    do_lower_case=False,
    keep_accents=True,
    multiple_of=2048,
):
    print("Converting SentencePiece model to HF tokenizer...")

    tokenizer = AlbertTokenizer.from_pretrained(
        spm_model_path,
        do_lower_case=do_lower_case,
        keep_accents=keep_accents,
    )

    n_to_add = 0
    if len(tokenizer) % multiple_of != 0:
        n_to_add = multiple_of - (len(tokenizer) % multiple_of)

    if n_to_add > 0:
        special_tokens = [f"[XXXXX{i}]" for i in range(n_to_add)]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )

    print(f"{n_to_add} tokens added.")
    print(
        f"Final vocab size: {len(tokenizer)} "
        "(set this in the LM config)."
    )

    tokenizer.save_pretrained(output_dir)
    print(f"HF tokenizer saved to: {output_dir}")

# ============================================================
# PIPELINE FOR ONE TOKENIZER
# ============================================================
def build_tokenizer(lang1, lang2, condition):
    l1_code = LANG_CODES[lang1]
    l2_code = LANG_CODES[lang2]

    tokenizer_name = f"bilingual_{lang1}_{lang2}_{condition}"
    print(f"\n=================================================")
    print(f"Building tokenizer: {tokenizer_name}")
    print(f"Languages: {l1_code}, {l2_code}")
    print("=================================================\n")

    f1 = file_for_lang(l1_code)
    f2 = file_for_lang(l2_code)

    merged = f"tmp_{tokenizer_name}.txt"

    # --------------------------------------------------------
    # Merge corpora
    # --------------------------------------------------------
    print(f"Merging into: {merged}")
    with open(merged, "w", encoding="utf-8") as out:
        ds = load_dataset(REPO_ID, split="train", streaming=True)
        for row in ds:
            text = row.get("text") or row.get("content")
            file_path = row.get("file") or row.get("path")
            if not (text and file_path):
                continue
            if file_path.endswith(f1) or file_path.endswith(f2):
                out.write(text.rstrip("\n") + "\n")

    # --------------------------------------------------------
    # Train SentencePiece
    # --------------------------------------------------------
    spm_prefix = os.path.join(SPM_DIR, tokenizer_name)
    print("Training SentencePiece...")
    spm.SentencePieceTrainer.train(
        input=merged,
        model_prefix=spm_prefix,
        vocab_size=VOCAB_SIZE,
        input_sentence_size=999_999_999,
        train_extremely_large_corpus=True,
        shuffle_input_sentence=True,
        num_threads=16,
    )

    print(f"SPM model saved: {spm_prefix}.model")

    # --------------------------------------------------------
    # Convert to HF tokenizer
    # --------------------------------------------------------
    hf_path = os.path.join(HF_DIR, tokenizer_name)
    os.makedirs(hf_path, exist_ok=True)

    convert_spm_to_hf(
        spm_model_path=f"{spm_prefix}.model",
        output_dir=hf_path,
        do_lower_case=False,
        keep_accents=True,
        multiple_of=2048,
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------
    os.remove(merged)
    print(f"Removed temporary file {merged}")

# ============================================================
# RUN ALL TOKENIZERS
# ============================================================
for lang1, lang2, condition in BGPT_MODELS:
    build_tokenizer(lang1, lang2, condition)

print("\nAll bilingual tokenizers built successfully.")

