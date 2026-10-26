from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from load_data import load_train, split_train_val, load_test
from evaluate import evaluate
import numpy as np
import pandas as pd
import torch
import argparse
import os

MODEL_CONFIGS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}

SYSTEM_PROMPT = "You are an expert annotator of sarcasm in social media posts."
QUESTION = (
    "Is the following tweet sarcastic? A tweet is sarcastic if the author says something "
    "but means the opposite, typically to mock, criticise or be ironic.\n"
    "Answer with only 'yes' or 'no'.\n\nTweet: {tweet}"
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), required=True)
parser.add_argument("--shots", type=int, default=0,
                    help="number of examples in the prompt (0 = zero-shot)")
parser.add_argument("--seed", type=int, default=42,
                    help="seed for choosing the few-shot examples")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--load-4bit", action="store_true",
                    help="4-bit quantization (needs CUDA GPU)")
parser.add_argument("--limit", type=int, default=None,
                    help="only use the first N tweets (for testing)")
cli_args = parser.parse_args()

MODEL_KEY = cli_args.model
MODEL_NAME = MODEL_CONFIGS[MODEL_KEY]
RUN_NAME = f"{MODEL_KEY}_{cli_args.shots}shot"
print(f"Running {MODEL_NAME} with {cli_args.shots} shots, seed {cli_args.seed}")

training_data = load_train("./data/train.En.csv")
train_data, val_data = split_train_val(training_data)
test_data_A, _ = load_test("./data/task_A_En_test.csv",
                           "./data/task_B_En_test.csv")
test_data_A = test_data_A.rename(columns={"text": "tweet"})
if cli_args.limit:
    val_data = val_data.head(cli_args.limit)
    test_data_A = test_data_A.head(cli_args.limit)

examples = []
if cli_args.shots > 0:
    n_pos = cli_args.shots // 2
    pos = train_data[train_data["sarcastic"] == 1].sample(
        n_pos, random_state=cli_args.seed)
    neg = train_data[train_data["sarcastic"] == 0].sample(
        cli_args.shots - n_pos, random_state=cli_args.seed)
    examples = pd.concat([pos, neg]).sample(
        frac=1, random_state=cli_args.seed)
    examples = list(zip(examples["tweet"], examples["sarcastic"]))


def build_prompt(tweet):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_tweet, ex_label in examples:
        messages.append(
            {"role": "user", "content": QUESTION.format(tweet=ex_tweet)})
        messages.append(
            {"role": "assistant", "content": "yes" if ex_label == 1 else "no"})
    messages.append({"role": "user", "content": QUESTION.format(tweet=tweet)})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_kwargs = {"dtype": torch.float16 if device != "cpu" else torch.float32}
if cli_args.load_4bit:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs["device_map"] = "auto"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
if not cli_args.load_4bit:
    model.to(device)
model.eval()


def first_token_ids(words):
    return sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in words})


YES_IDS = first_token_ids(["yes", "Yes", " yes", " Yes"])
NO_IDS = first_token_ids(["no", "No", " no", " No"])


@torch.no_grad()
def predict_proba(tweets):
    # returns P(yes) / (P(yes) + P(no)) for each tweet
    probs = []
    for i in range(0, len(tweets), cli_args.batch_size):
        prompts = [build_prompt(t) for t in tweets[i:i + cli_args.batch_size]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        logits = model(**inputs).logits[:, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        yes = torch.logsumexp(log_probs[:, YES_IDS], dim=-1)
        no = torch.logsumexp(log_probs[:, NO_IDS], dim=-1)
        probs.extend(torch.sigmoid(yes - no).cpu().tolist())
        print(
            f"  {min(i + cli_args.batch_size, len(tweets))}/{len(tweets)}", end="\r")
    print()
    return np.array(probs)


print("Example prompt:\n" + build_prompt(test_data_A["tweet"].iloc[0]))

val_proba = predict_proba(val_data["tweet"].tolist())
test_proba = predict_proba(test_data_A["tweet"].tolist())

val_results = evaluate(val_data["sarcastic"], (val_proba > 0.5).astype(int))
test_results = evaluate(
    test_data_A["sarcastic"], (test_proba > 0.5).astype(int))
print(f"Validation: {val_results}")
print(f"Test: {test_results}")

thresholds = np.unique(np.quantile(val_proba, np.linspace(0.01, 0.99, 99)))
best_threshold = max(thresholds, key=lambda t: evaluate(
    val_data["sarcastic"], (val_proba > t).astype(int))["f1_sarcastic"])
val_tuned = evaluate(val_data["sarcastic"],
                     (val_proba > best_threshold).astype(int))
test_tuned = evaluate(test_data_A["sarcastic"],
                      (test_proba > best_threshold).astype(int))
print(f"Tuned threshold {best_threshold:.3f} -> validation: {val_tuned}")
print(f"Tuned threshold {best_threshold:.3f} -> test: {test_tuned}")

if cli_args.limit:
    print("--limit was used, not saving results")
else:
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({
        "text": test_data_A["tweet"],
        "sarcastic": test_data_A["sarcastic"],
        "pred": (test_proba > 0.5).astype(int),
        "pred_tuned": (test_proba > best_threshold).astype(int),
        "p_sarcastic": test_proba,
    }).to_csv(f"results/{RUN_NAME}_seed{cli_args.seed}_test_predictions.csv", index=False)

    metrics = pd.DataFrame([
        {"model": RUN_NAME, "seed": cli_args.seed,
            "split": "validation", **val_results},
        {"model": RUN_NAME, "seed": cli_args.seed, "split": "test", **test_results},
        {"model": f"{RUN_NAME}_tuned", "seed": cli_args.seed,
            "split": "validation", **val_tuned},
        {"model": f"{RUN_NAME}_tuned", "seed": cli_args.seed,
            "split": "test", **test_tuned},
    ])
    metrics.to_csv(
        f"results/{RUN_NAME}_seed{cli_args.seed}_metrics.csv", index=False)
