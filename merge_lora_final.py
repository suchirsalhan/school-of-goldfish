import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    print("🔹 Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    print("🔹 Loading LoRA adapters")
    model = PeftModel.from_pretrained(
        base,
        args.lora_repo,
        revision=args.revision,
    )

    print("🔹 Merging LoRA → full weights")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model → {args.out_dir}")
    model.save_pretrained(args.out_dir)

    tok = AutoTokenizer.from_pretrained(args.lora_repo, revision=args.revision)
    tok.save_pretrained(args.out_dir)

    print("✅ Done")

if __name__ == "__main__":
    main()
