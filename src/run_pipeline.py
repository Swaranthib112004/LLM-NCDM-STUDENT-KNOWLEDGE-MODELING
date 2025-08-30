import os, json
from llm_concepts import run as concepts_run
from sbert_optimize_q import run as sbert_q_run
from convert_to_neuralcdm_json import run as json_run
from ncdm_train import train_ncdm

def main():
    os.makedirs("reports", exist_ok=True)
    print("1) Extracting concepts + Bloom with LLaMA (Ollama)...")
    concepts_run("data/items.csv", "reports")

    print("2) Building & optimizing Q-matrix with SBERT...")
    opt_q = sbert_q_run("data/items.csv", "reports")

    print("3) Converting responses -> JSON (if needed by other tools)...")
    json_run("data/responses.csv", "data/ncdm_json")

    print("4) Training simple NCDM...")
    rmse, auc, acc, f1, prec, rec = train_ncdm("data/responses.csv", opt_q, "reports", epochs=10, lr=0.2)

    print("✅ Metrics:")
    print(f"   RMSE={rmse:.3f}")
    print(f"   AUC={auc}")
    print(f"   Accuracy={acc:.3f}")
    print(f"   F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

if __name__ == "__main__":
    main()
