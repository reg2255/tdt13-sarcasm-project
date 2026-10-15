from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(y_true, y_pred):
    #returns dict wit f1 score, f1 macro and precision and recall scores
    f1 = f1_score(y_true, y_pred, pos_label=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1)
    return {
        "f1_sarcastic": f1,
        "f1_macro": macro_f1,
        "precision": precision,
        "recall": recall
    }

if __name__ == "__main__":
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    print(evaluate(y_true, y_pred))