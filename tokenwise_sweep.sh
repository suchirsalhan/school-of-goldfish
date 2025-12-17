#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

declare -A MODELS=(
  [es]="spa_latn_1000mb catherinearnett/B-GPT_en_es_simultaneous"
  [el]="ell_grek_1000mb catherinearnett/B-GPT_en_el_simultaneous"
  [nl]="nld_latn_1000mb catherinearnett/B-GPT_en_nl_simultaneous"
  [pl]="pol_latn_1000mb catherinearnett/B-GPT_en_pl_simultaneous"
)

HF_USER="suchirsalhan"

# Sweep through alpha_l1 interpolation values
ALPHAS=(0.1 0.25 0.33 0.5 0.66 0.75 0.9)

for L2 in es el nl pl; do
  read L2_MODEL TOKENIZER <<< "${MODELS[$L2]}"

  if [ "$L2" = "el" ]; then
    LANG_PAIR="el-en"
  else
    LANG_PAIR="en-$L2"
  fi

  BASE_OUTPUT_DIR="out/en_${L2}_tokenwise"

  for ALPHA in "${ALPHAS[@]}"; do
    # Convert alpha to integer percentage for folder naming
    PERCENT=$(awk "BEGIN {printf \"%d\", $ALPHA*100}")

    # Append alpha fraction to output folder
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/alpha${PERCENT}"

    echo "==== Running tokenwise merge for $L2 with alpha_l1=$ALPHA ($PERCENT%) ===="

    python bilingual_tokenwise_merge.py \
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
      --tokenizer_aware_merge \
      --alpha_l1 $ALPHA \
      --push_hf
  done
done
