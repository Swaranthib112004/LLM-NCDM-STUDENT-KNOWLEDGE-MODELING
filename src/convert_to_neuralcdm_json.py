import os, json
import pandas as pd

def run(input_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    out = []
    for _, row in df.iterrows():
        out.append({
            "user_id": str(row["student_id"]),
            "item_id": str(row["item_id"]),
            "is_correct": int(row["correct"]),
        })
    out_path = os.path.join(output_dir, "responses.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✅ Converted {len(out)} records to {out_path}")
    return out_path
