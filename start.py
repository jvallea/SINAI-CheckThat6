import os
from datetime import datetime


tasks = ["PR2","HN", "FC", "RD", "C19"]
models = ["BiLSTM", "surprise"]
search_algs = ["shap", "bf", "memory_bf", "keybert", "shap_hybrid", "keybert_hybrid"]
heuristics = ["homo", "inv", "mix", "syn"]

for t in tasks:
    for m in models:
        for s in search_algs:
            if t in ["HN", "RD"] and s in ["shap"]: continue #SHAP gives an error with large texts

            for h in heuristics:
                    print(f"[{datetime.now()}] ---///  {t}-{m}-{s}-{h}   ///---")
                    os.system(f"python3 -u content/BODEGA/adversarial.py --t {t} --v {m} --s {s} --h {h} > content/BODEGA/outputs/{t}-{m}-{s}-{h}.log")
