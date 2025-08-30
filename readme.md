# LLM + SBERT + NCDM (Local Ollama)

## 1) Install
```bash
python -m venv .venv
.venv\Scripts\activate     # on Windows
pip install -r requirements.txt
```

## 2) Pull a local LLaMA model (Ollama)
Install Ollama (Windows installer) and then in a terminal:
```bash
ollama pull llama3:3b
# or for larger:
ollama pull llama3:7b
```
Optionally set model:
```bash
set LLM_MODEL=llama3:3b
```

## 3) Run pipeline
```bash
python src/run_pipeline.py
```

This creates files in `reports/`: 
- `questions_bloom.csv`
- `concepts_per_item.json`
- `concept_universe.json`
- `q_matrix_raw.csv`
- `q_matrix_optimized.csv`
- `training_metrics.csv`
- `student_mastery.csv`
- `item_difficulty.csv`
- `metrics.json`

## 4) Launch Streamlit dashboard
```bash
streamlit run src/app_streamlit.py
```

Notes:
- If Ollama is not running, concept extraction falls back to simple keyword rules.
- SBERT optimization uses `all-MiniLM-L6-v2`. If it can't load, a keyword overlap fallback is used.
- NCDM trainer here is a **simple NumPy version** (sigmoid of mastery·Q - difficulty). It’s lightweight and CPU-friendly
