# tdt13-sarcasm-project

Kan kjøre fra terminal. 
`python src/train_transformer.py --model <roberta, bertweet or modernbert> --seed <choose seed>` or run all at the same time with `caffeinate -i bash -c 'for model in roberta bertweet modernbert; do for seed in <seeds you want with space between them>; do python src/train_transformer.py --model $model --seed $seed; done; done'`