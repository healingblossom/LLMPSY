

import yaml
import json
import os

from model_manager import ModelManager
from prompt_formatter import build_messages
from prompts.task_prompts import get_prompt
from utils.data_loader import load_test_data
from utils.result_saver import save_results


def run_each_task(
    model, 
    model_config, 
    task_def, 
    variant_def, 
    global_config,
    previous_results=None
):
    """
    Führt eine einzelne Task mit einer Prompt-Variante für ein Modell aus.
    
    Args:
        model: Modell-Objekt (hat .generate() Methode)
        model_config: Modell-Konfiguration aus config.yaml
        task_def: Task-Definition aus tasks.yaml
        variant_def: Prompt-Varianten-Definition
        global_config: Globale Konfiguration
        previous_results: Optionale Ergebnisse aus vorheriger Task
    
    Returns:
        List of result dicts
    """
    
    task_id = task_def["id"]
    variant_id = variant_def["id"]
    template_id = variant_def["template"]
    
    print(f"\n  Task: {task_id}, Variante: {variant_id}")
    
    # 1. Lade Prompt-Template
    prompt_template = get_prompt(template_id)
    if not prompt_template:
        print(f"  Fehler: Prompt-Template '{template_id}' nicht gefunden")
        return []
    
    system_text = prompt_template["system"]
    user_template = prompt_template["user_template"]
    
    # 2. Lade Eingabedaten
    if list(filter(lambda input: input.startswith("results_of") , task_def["input_file"])): # existieren einträge in input_files, die results früherer tasks sind
        # Input aus vorheriger Task
        if previous_results is None:
            print(f"  Fehler: Keine vorherigen Ergebnisse für {task_def['input_file']}")
            return []
        input_data = previous_results
    else:
        # Input aus CSV/Datei
        input_file = os.path.join(
            global_config["paths"]["unsafe_directory"],
            "input_data",
            task_def["input_file"]
        )
        input_data = load_test_data(input_file)
    
    # 3. Führe Model für jede Eingabe aus
    results = []
    for case in input_data:
        # Baue User-Text
        if "transcript" in user_template:
            user_text = user_template.format(transcript=case.get("text", ""))
        elif "symptoms" in user_template:
            user_text = user_template.format(symptoms=json.dumps(case.get("symptoms", [])))
        elif "diagnosis" in user_template:
            user_text = user_template.format(
                diagnosis=case.get("diagnosis", ""),
                symptoms=json.dumps(case.get("symptoms", []))
            )
        else:
            user_text = user_template
        
        # Baue Messages
        messages = build_messages(system_text, user_text, model_config) # Funktion aus prompt_formatter
        
        # Generiere Output
        try:
            response = model.generate_from_messages(messages)
            parsed = parse_json_output(response)
        except Exception as e:
            print(f"    Fehler bei Case {case.get('id', '?')}: {e}")
            parsed = {"error": str(e)}
        
        results.append({
            "case_id": case.get("id", "unknown"),
            "output": parsed,
        })
    
    return results


def parse_json_output(response_text: str):
    """Versucht, JSON aus Modell-Antwort zu extrahieren"""
    response_text = response_text.strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: Suche nach [ ... ]
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response_text[start:end+1])
            except json.JSONDecodeError:
                return {"raw": response_text[:200]}
        return {"raw": response_text[:200]}


def run_all_tasks(model_name, selected_tasks=None, selected_variants=None):
    """
    Führt alle Tasks für ein Modell aus.
    
    Args:
        model_name: Name des Modells aus config.yaml
        selected_tasks: Optional: nur diese Task-IDs ausführen
        selected_variants: Optional: nur diese Prompt-Varianten-IDs ausführen
    """
    

    # Lade Configs
    global_config = yaml.safe_load(open("config/config.yaml"))
    tasks_config = yaml.safe_load(open("config/tasks.yaml"))

    # Lade Modell
    manager = ModelManager("config/config.yaml")
    model_spec = manager.get_model_spec(model_name)
    model = manager.load_model(model_name)
    
    # Speichere Ergebnisse pro Task
    task_results = {}
    
    # Führe Tasks in Reihenfolge aus
    for task in tasks_config["tasks"]:
        task_id = task["id"]
        
        if selected_tasks and task_id not in selected_tasks:
            continue
        
        print(f"\nTask: {task_id}")
        
        # Lade vorherige Task-Ergebnisse falls nötig
        previous_results = None
        if task["input_file"].startswith("results_of"):
            prev_task_id = task["input_file"].replace("results_of", "")
            if prev_task_id in task_results:
                previous_results = task_results[prev_task_id]
        
        # Führe jede Prompt-Variante aus
        for variant_def in task["prompt_variants"]:
            variant_id = variant_def["id"]
            
            if selected_variants and variant_id not in selected_variants:
                continue
            
            results = run_each_task(
                model,
                model_spec,
                task,
                variant_def,
                global_config,
                previous_results
            )
            
            # Speichere Ergebnisse
            task_key = f"{task_id}_{variant_id}"
            task_results[task_key] = results  # Für nächste Task
            
            save_results(
                global_config,
                task_id,
                variant_id,
                model_name,
                results
            )
    
    # Cleanup
    manager.cleanup_model(model_name)
    print(f"\nModell {model_name} fertig!")
