import os, json, re
import pandas as pd

USE_OLLAMA = True
try:
    import ollama  # requires Ollama app running
except Exception as e:
    USE_OLLAMA = False

BLOOM_LEVELS = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]

SYSTEM_PROMPT = (
    "You are an expert in educational assessment. "
    "Given a question, extract 2–4 short concept tags and 1 Bloom level. "
    "Return STRICT JSON with fields: concepts (list of strings), bloom (one of Remember, Understand, Apply, Analyze, Evaluate, Create)."
)

def _fallback_concepts(text: str):
    text_l = text.lower()
    concepts = []
    if "stack" in text_l: concepts += ["Stack", "LIFO"]
    if "queue" in text_l: concepts += ["Queue", "FIFO"]
    if "binary search" in text_l: concepts += ["Binary Search", "LogN"]
    if "binary tree" in text_l: concepts += ["Binary Tree", "Traversal"]
    if "inorder" in text_l: concepts += ["Inorder Traversal"]
    if "cycle" in text_l: concepts += ["Cycle Detection", "Floyd"]
    if "linked list" in text_l: concepts += ["Linked List"]
    if "undo" in text_l: concepts += ["Stack"]
    if "graph" in text_l and "level" in text_l: concepts += ["BFS", "Graph Traversal"]
    if "postfix" in text_l: concepts += ["Postfix", "Stacks"]
    if "time complexity" in text_l: concepts += ["Complexity"]
    bloom = "Understand"
    if any(k in text_l for k in ["use", "find", "convert"]): bloom = "Apply"
    if any(k in text_l for k in ["compare", "which"]): bloom = "Analyze"
    return {"concepts": sorted(set(concepts))[:4], "bloom": bloom}

def ask_llama(question: str):
    prompt = (
        SYSTEM_PROMPT + "\n"
        + "Question: " + question + "\n"
        + "Respond ONLY JSON. Example: {\"concepts\":[\"Stack\",\"LIFO\"],\"bloom\":\"Understand\"}"
    )
    if USE_OLLAMA:
        try:
            resp = ollama.chat(model=os.getenv("LLM_MODEL","llama3:3b"), messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])
            txt = resp["message"]["content"]
            j = json.loads(txt)
            if "concepts" in j and "bloom" in j:
                return j
        except Exception as e:
            pass
    # fallback
    return _fallback_concepts(question)

def run(items_csv: str, out_dir: str = "reports"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(items_csv)
    out_rows = []
    all_concepts = set()

    for _, row in df.iterrows():
        qid = row["item_id"]
        text = row["question_text"]
        ans = ask_llama(text)
        concepts = ans.get("concepts", [])
        bloom = ans.get("bloom", "Understand")
        all_concepts.update(concepts)
        out_rows.append({"item_id": qid, "question_text": text, "concepts": ";".join(concepts), "bloom": bloom})

    result_df = pd.DataFrame(out_rows)
    result_df.to_csv(os.path.join(out_dir, "questions_bloom.csv"), index=False)

    # also save raw concept list per item
    concepts_map = {r["item_id"]: r["concepts"].split(";") if r["concepts"] else [] for r in out_rows}
    with open(os.path.join(out_dir, "concepts_per_item.json"), "w") as f:
        json.dump(concepts_map, f, indent=2)

    # save concept universe
    with open(os.path.join(out_dir, "concept_universe.json"), "w") as f:
        json.dump(sorted(c for c in all_concepts if c), f, indent=2)

    print(f"✅ Concepts + Bloom saved to {out_dir}/questions_bloom.csv and concepts_per_item.json")
    return os.path.join(out_dir, "questions_bloom.csv")
