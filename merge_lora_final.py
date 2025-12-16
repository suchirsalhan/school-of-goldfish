#!/usr/bin/env python
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_checkpoint(base_model, lora_repo, revision, out_dir):
    print(f"🔹 Merging {lora_repo} / {revision} → {out_dir}")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, lora_repo, revision=revision)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained(lora_repo, revision=revision)
    tok.save_pretrained(out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_dir", required=True, help="Directory containing checkpoint-* subdirs")
    parser.add_argument("--hf_user", required=True)
    parser.add_argument("--repo_name", required=True)
    args = parser.parse_args()

    base_model = args.base_model
    lora_dir = Path(args.lora_dir)
    hf_user = args.hf_user
    repo_name = args.repo_name

    # Sort checkpoints by name (natural sort)
    checkpoints = sorted([d for d in lora_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

    merged_dir = lora_dir / "merged"
    merged_dir.mkdir(exist_ok=True)

    final_ckpt = checkpoints[-1]

    for ckpt in checkpoints:
        out_subdir = merged_dir / ckpt.name
        merge_checkpoint(base_model, str(ckpt), revision="main", out_dir=str(out_subdir))

    # The final checkpoint will also be the 'main' branch
    final_out = merged_dir / final_ckpt.name / "main"
    final_out.parent.mkdir(parents=True, exist_ok=True)
    final_out.rename(merged_dir / "main")

    print(f"✅ All checkpoints merged. Final checkpoint → main")

if __name__ == "__main__":
    main()
