import os, json, math, csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, precision_score, recall_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_ncdm(responses_csv: str, q_matrix_csv: str, reports_dir: str = "reports", epochs: int = 10, lr: float = 0.2):
    os.makedirs(reports_dir, exist_ok=True)

    R = pd.read_csv(responses_csv)  # student_id,item_id,correct
    Q = pd.read_csv(q_matrix_csv)   # item_id + concept columns
    concepts = [c for c in Q.columns if c != "item_id"]

    # index maps
    students = sorted(R["student_id"].unique())
    items = Q["item_id"].tolist()
    s_index = {s:i for i,s in enumerate(students)}
    i_index = {it:i for i,it in enumerate(items)}

    # build interaction matrix triples
    triples = []
    for _, r in R.iterrows():
        s = s_index[r["student_id"]]
        i = i_index.get(r["item_id"])
        if i is None: 
            continue
        y = int(r["correct"])
        triples.append((s,i,y))

    n_students = len(students)
    n_items = len(items)
    n_concepts = len(concepts)

    # parameters
    S = np.random.rand(n_students, n_concepts) * 0.2 + 0.4  # initial mastery ~0.5
    D = np.zeros(n_items)  # item difficulty
    Qm = Q[concepts].values.astype(float)

    # training
    losses = []
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for (s,i,y) in triples:
            z = (S[s] * Qm[i]).sum() - D[i]         # mastery*concepts - difficulty
            p = sigmoid(z)
            # BCE loss
            loss = -(y * math.log(p+1e-9) + (1-y) * math.log(1-p+1e-9))
            total_loss += loss

            # gradients
            grad = (p - y)
            # dS for active concepts only
            S[s] -= lr * grad * Qm[i]
            # difficulty
            D[i] += lr * grad

        avg_loss = total_loss / max(1, len(triples))
        losses.append(avg_loss)

    # Evaluate
    y_true, y_prob = [], []
    for (s,i,y) in triples:
        z = (S[s] * Qm[i]).sum() - D[i]
        p = sigmoid(z)
        y_true.append(y)
        y_prob.append(p)
    y_pred = [1 if p>=0.5 else 0 for p in y_prob]

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_prob))
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # save reports
    pd.DataFrame({"epoch": list(range(1,epochs+1)), "loss": losses}).to_csv(os.path.join(reports_dir, "training_metrics.csv"), index=False)
    sm = pd.DataFrame(S, columns=concepts)
    sm.insert(0,"student_id", students)
    sm.to_csv(os.path.join(reports_dir, "student_mastery.csv"), index=False)

    idf = pd.DataFrame({"item_id": items, "difficulty": D})
    idf.to_csv(os.path.join(reports_dir, "item_difficulty.csv"), index=False)

    metrics = {"RMSE": rmse, "AUC": auc, "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec}
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Training complete! RMSE={rmse:.3f}, AUC={auc if isinstance(auc,float) else auc}, Acc={acc:.3f}, F1={f1:.3f}")
    return rmse, auc, acc, f1, prec, rec
