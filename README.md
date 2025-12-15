# Glints of Gold or Murky Waters? Can a School of Merged Monolingual Goldfish Models Swim in Multilingual Waters?

Merging Goldfish:
```
python bilingual_merge_and_eval.py \
  --l1 en --l2 es \
  --model_l1 goldfish-models/eng_latn_1000mb \
  --model_l2 goldfish-models/spa_latn_1000mb \
  --bilingual_tokenizer catherinearnett/B-GPT_en_es_simultaneous \
  --output_dir out/en_es_merge \
  --do_merge \
  --do_train \
  --do_eval
```

Just to evaluate B-GPT
```
python bilingual_merge_and_eval.py \
  --l1 en --l2 es \
  --bgpt_model catherinearnett/B-GPT_en_es_simultaneous \
  --do_eval
```
