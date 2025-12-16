#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

declare -A MODELS=(
  [es]="spa_latn_1000mb catherinearnett/B-GPT_en_es_simultaneous"
  [el]="ell_grek_1000mb catherinearnett/B-GPT_en_el_simultaneous"
  [nl]="nld_latn_1000mb catherinearnett/B-GPT_en_nl_simultaneous"
  [pl]="pol_latn_1000mb catherinearnett/B-GPT_en_pl_simultaneous"
)

for L2 in es el nl pl; do
  read L2_MODEL TOKENIZER <<< "${MODELS[$L2]}"

  python bilingual_merge_and_eval.py \
    --l1 en --l2 $L2 \
    --model_l1 goldfish-models/eng_latn_1000mb \
    --model_l2 goldfish-models/$L2_MODEL \
    --bilingual_tokenizer $TOKENIZER \
    --dataset opus_books \
    --lang_pair en-$L2 \
    --output_dir out/en_${L2}_merge \
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
    --max_eval_examples 2000
done
