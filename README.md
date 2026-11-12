# Detecting Intended Sarcasm in Tweets: Comparing Fine-Tuned Encoders and a Large Language Model
This is the code for the experiments in the TDT13 sarcasm detection project.

## BERT models
Can run one by on in terminal:
- `python src/train_transformer.py --model <roberta, bertweet or modernbert> --seed <choose seed>` 

Or run all at the same time with:
- `caffeinate -i bash -c 'for model in roberta bertweet modernbert; do for seed in <seeds you want with space between them>; do python src/train_transformer.py --model $model --seed $seed; done; done'`

## LLM (Qwen2.5-7B-Instruct)
Run in Google Colab with a T4 GPU, with the project folder in MyDrive/tdt13-sarscasm-project.
- Zero-/few-shot prompting: `python src/llm_prompt.py --model qwen7b --shots <0 or 10> --seed <seed> --load-4bit`
- LoRA fine-tuning (4-bit): `python src/train_transformer.py --model qwen7b_lora --seed <seed>`
