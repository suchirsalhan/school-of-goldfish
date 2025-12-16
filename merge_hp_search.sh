#!/usr/bin/env bash
# Hyperparameter search for bilingual merge (Stage 1)
# - Merge + small evaluation set (100–500 examples)
# - No fine-tuning
# - Save all metrics to CSV
# - Save merged checkpoints for each combination
# - Push each checkpoint to Hugging Face

set -euo pipefail

# ----------------------------
# Search space
# ----------------------------
start_weights=(0.5 0.7 0.9)
end_weights=(0.1 0.3 0.5)
lora_ranks=(8 16 32)
lora_alphas=(16 32 64)
freeze_ratios=(0.3 0.5 0.7)

L2_LANGS=("es" "el" "nl" "pl")

declare -A MODELS=(
    ["es"]="spa_latn_1000mb:catherinearnett/B-GPT_en_es_simultaneous"
    ["el"]="ell_grek_1000mb:catherinearnett/B-GPT_en_el_simultaneous"
    ["nl"]="nld_latn_1000mb:catherinearnett/B-GPT_en_nl_simultaneous"
    ["pl"]="pol_latn_1000mb:catherinearnett/B-GPT_en_pl_simultaneous"
)

MAX_EVAL=500
CSV_FILE="merge_hp_search_results.csv"

# ----------------------------
# Ensure output dir
# ----------------------------
mkdir -p out/hp_search

# ----------------------------
# Write CSV header
# ----------------------------
echo "l2,start_weight,end_weight,lora_rank,lora_alpha,freeze_ratio,ppl,fragmentation_l1,fragmentation_l2,entropy_l1,entropy_l2,shared_tokens,l1_only_tokens,l2_only_tokens,checkpoint_dir,hf_repo_name" > "$CSV_FILE"

# ----------------------------
# Iterate over languages and hyperparameters
# ----------------------------
for L2 in "${L2_LANGS[@]}"; do
    IFS=":" read -r L2_MODEL TOKENIZER <<< "${MODELS[$L2]}"
    if [[ "$L2" == "el" ]]; then
        LANG_PAIR="el-en"
    else
        LANG_PAIR="en-$L2"
    fi

    for start_w in "${start_weights[@]}"; do
    for end_w in "${end_weights[@]}"; do
    for r in "${lora_ranks[@]}"; do
    for alpha in "${lora_alphas[@]}"; do
    for freeze in "${freeze_ratios[@]}"; do

        hf_name="${LANG_PAIR}_w${start_w}-${end_w}_r${r}_a${alpha}_f${freeze}"
        out_dir="out/hp_search/$hf_name"
        mkdir -p "$out_dir"

        echo "Running merge and eval for $hf_name..."

        python bilingual_merge_and_eval.py \
            --l1 en --l2 "$L2" \
            --model_l1 goldfish-models/eng_latn_1000mb \
            --model_l2 "goldfish-models/$L2_MODEL" \
            --bilingual_tokenizer "$TOKENIZER" \
            --dataset opus_books \
            --lang_pair "$LANG_PAIR" \
            --do_merge \
            --do_eval \
            --output_dir "$out_dir" \
            --max_eval_examples "$MAX_EVAL" \
            --lora_rank "$r" \
            --lora_alpha "$alpha" \
            --freeze_ratio "$freeze" \
            --start_weight "$start_w" \
            --end_weight "$end_w" \
            --push_hf False

        # Append results to CSV if file exists
        run_csv="$out_dir/bilingual_diagnostics.csv"
        if [[ -f "$run_csv" ]]; then
            awk -v l2="$L2" -v sw="$start_w" -v ew="$end_w" -v r="$r" -v a="$alpha" -v fz="$freeze" -v dir="$out_dir" -v hf="$hf_name" \
                'BEGIN{FS=OFS=","} NR>1{$(NF+1)=l2; $(NF+1)=sw; $(NF+1)=ew; $(NF+1)=r; $(NF+1)=a; $(NF+1)=fz; $(NF+1)=dir; $(NF+1)=hf; print}' \
                "$run_csv" >> "$CSV_FILE"
        fi

        # Push checkpoint to Hugging Face
        echo "Pushing $hf_name → Hugging Face"
        huggingface-cli repo create "$hf_name" --type model --yes || true
        git -C "$out_dir" init || true
        git -C "$out_dir" add .
        git -C "$out_dir" commit -m "Merged checkpoint $hf_name" || true
        git -C "$out_dir" branch -M main || true
        git -C "$out_dir" remote add origin "https://huggingface.co/suchirsalhan/$hf_name" || true
        git -C "$out_dir" push -u origin main --force || true

    done
    done
    done
    done
    done
done

echo "Hyperparameter search done. Results saved → $CSV_FILE"


