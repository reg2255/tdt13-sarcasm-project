from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from load_data import load_train, split_train_val, load_test
import numpy as np
from evaluate import evaluate

#to convert pandas table to Hugging Face-dataset
def to_dataset(df, tokenizer):
    df = df[["tweet", "sarcastic"]].rename(columns={"sarcastic": "labels"})
    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(batch["tweet"], truncation=True, max_length=128),
        batched=True
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return evaluate(labels, preds)

MODEL_NAME = "roberta-base"
SEED = 42
 
training_data = load_train("./data/train.En.csv")
train_data, val_data = split_train_val(training_data)
test_data_A, test_data_B = load_test("./data/task_A_En_test.csv", "./data/task_B_En_test.csv")
# do i need these?
test_data_A = test_data_A.rename(columns={"text": "tweet"})
test_data_B = test_data_B.rename(columns={"text": "tweet"})

# y_train = train_data["sarcastic"]
# y_val = val_data["sarcastic"]
y_test = test_data_A["sarcastic"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = to_dataset(train_data, tokenizer)
val_dataset = to_dataset(val_data, tokenizer)
test_dataset = to_dataset(test_data_A, tokenizer)

print(train_dataset)
print(train_dataset[0])

set_seed(SEED)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir="runs/roberta",
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_sarcastic",
    seed=SEED,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)
trainer.train()