#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# Monolingual L2 models
declare -A MONOS=(
  [es]="goldfish-models/spa_latn_1000mb"
  [el]="goldfish-models/ell_grek_1000mb"
  [nl]="goldfish-models/nld_latn_1000mb"
  [pl]="goldfish-models/pol_latn_1000mb"
)

# Pick a representative bilingual tokenizer (L2->en simultaneous) for each L2
declare -A TOKENIZERS=(
  [es]="catherinearnett/B-GPT_es_en_simultaneous"
  [el]="catherinearnett/B-GPT_el_en_simultaneous"
  [nl]="catherinearnett/B-GPT_nl_en_simultaneous"
  [pl]="catherinearnett/B-GPT_pl_en_simultaneous"
)

HF_USER="suchirsalhan"

# Top-layer fractions to sweep
MERGE_FRACS=(0.1 0.2 0.25 0.33 0.5 0.66 0.75,1.0)

for L2 in es el nl pl; do
  L2_MODEL="${MONOS[$L2]}"
  TOKENIZER="${TOKENIZERS[$L2]}"

  if [ "$L2" = "el" ]; then
    LANG_PAIR="el-en"
  else
    LANG_PAIR="en-$L2"
  fi

  BASE_OUTPUT_DIR="out/en_${L2}_merge"

  for FRAC in "${MERGE_FRACS[@]}"; do
    # Convert fraction to integer percentage for folder naming
    PERCENT=$(awk "BEGIN {printf \"%d\", $FRAC*100}")

    # Append top fraction to output folder
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/top${PERCENT}"

    echo "==== Running merge for $L2 with top fraction $FRAC ($PERCENT%) ===="

    python hyp-bilingual-merge.py \
      --l1 en --l2 $L2 \
      --model_l1 goldfish-models/eng_latn_1000mb \
      --model_l2 $L2_MODEL \
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
      --merge_mode top \
      --merge_top_frac "$FRAC" \
      --push_hf
  done
done

