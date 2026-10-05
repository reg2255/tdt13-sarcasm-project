from load_data import load_train, split_train_val, load_test
from evaluate import evaluate
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# train and validate on the model
training_data = load_train("./data/train.En.csv")
train_data, val_data = split_train_val(training_data)

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(train_data["tweet"])
x_val = vectorizer.transform(val_data["tweet"])

y_train = train_data["sarcastic"]

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(x_train, y_train)

y_val = val_data["sarcastic"]
val_pred = model.predict(x_val)
val_results = evaluate(y_val, val_pred)
print(f"Validation: {val_results}")

# test the model
test_data_A, test_data_B = load_test("./data/task_A_En_test.csv", "./data/task_B_En_test.csv")
x_test = vectorizer.transform(test_data_A["text"])

y_test = test_data_A["sarcastic"]
test_pred = model.predict(x_test)
test_results = evaluate(y_test, test_pred)
print(f"Test results: {test_results}")

# save results
os.makedirs("results", exist_ok=True)

test_predictions = pd.DataFrame({
    "text": test_data_A["text"],
    "sarcastic": y_test,
    "pred": test_pred
})
test_predictions.to_csv("results/baseline_test_predictions.csv", index=False)

metrics = pd.DataFrame([
    {"model": "baseline", "split": "validation", **val_results},
    {"model": "baseline", "split": "test", **test_results}
])
metrics.to_csv("results/baseline_metrics.csv", index=False)