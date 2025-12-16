#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Map of L2 languages to model/tokenizer
declare -A MODELS=(
  [es]="spa_latn_1000mb catherinearnett/B-GPT_en_es_simultaneous"
  [el]="ell_grek_1000mb catherinearnett/B-GPT_en_el_simultaneous"
  [nl]="nld_latn_1000mb catherinearnett/B-GPT_en_nl_simultaneous"
  [pl]="pol_latn_1000mb catherinearnett/B-GPT_en_pl_simultaneous"
)

HF_USER="suchirsalhan"  # your HF username

for L2 in es el nl pl; do
  read L2_MODEL TOKENIZER <<< "${MODELS[$L2]}"

  # Special-case Greek (OPUS direction issue)
  if [ "$L2" = "el" ]; then
    LANG_PAIR="el-en"
  else
    LANG_PAIR="en-$L2"
  fi

  OUTPUT_DIR="out/en_${L2}_merge"
  HF_REPO_MERGED="${HF_USER}/en_${L2}_merge-merged"
  HF_REPO_TRAINED="${HF_USER}/en_${L2}_merge-trained"

  # Create HF repos programmatically if not existing
  python - <<END
from huggingface_hub import create_repo

for repo in ["$HF_REPO_MERGED", "$HF_REPO_TRAINED"]:
    try:
        create_repo(repo, exist_ok=True)
        print(f"Repo ready: {repo}")
    except Exception as e:
        print(f"Repo exists or error: {repo} -> {e}")
END

  # Run the Python merge/train/eval script
  python bilingual_merge_and_eval.py \
    --l1 en --l2 $L2 \
    --model_l1 goldfish-models/eng_latn_1000mb \
    --model_l2 goldfish-models/$L2_MODEL \
    --bilingual_tokenizer $TOKENIZER \
    --dataset opus_books \
    --lang_pair $LANG_PAIR \
    --output_dir $OUTPUT_DIR \
    --do_merge \
    --do_train \
    --do_eval \
    --epochs 1 \
    --lr 1e-4 \
    --train_bs 16 \
    --grad_accum 2 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --freeze_ratio 0.6 \
    --max_train_examples 50000 \
    --max_eval_examples 2000 \
    --push_hf
done

