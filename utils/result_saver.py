# utils/result_saver.py speichert die Ergebnisse

import json
import os


def save_results(global_config, task_id, variant_id, model_name, results):
    """
    Speichert Evaluierungs-Ergebnisse als JSON.
    
    Args:
        global_config: Config dict mit Pfaden
        task_id: Task-ID (z.B. "task1_symptom_extraction")
        variant_id: Varianten-ID (z.B. "zeroshot")
        model_name: Modell-Name
        results: List of result dicts
    """
    
    results_dir = os.path.join(
        global_config["paths"]["safe_directory"],
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{task_id}_{variant_id}_{model_name}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"    Ergebnisse gespeichert: {filepath}")
