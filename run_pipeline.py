# Skript zum durchlaufen lassen beliebig vieler pipelines
# wird über "python run_pipeline.py" aufgerufen, es können beliebig viele namen von Modellen angehangen werden, die auch ausgeführt werden sollen bspw: "python run_pipeline.py mistral"

import sys
import yaml
from run_in_env import run_in_conda_env

print("\n" + "="*60)
print("STARTE LLM-EVALUIERUNG")
print("="*60)

try:
    # 0. Optional: Modelle aus Kommandozeile
    # Beispiel-Aufruf:
    #   python run_pipeline.py modellname1 modellname2
    selected_models = sys.argv[1:]  # alles nach dem Skriptnamen
    if selected_models:
        print(f"\nEs wurden Modelle explizit angegeben: {selected_models}")
    else:
        print("\nKeine Modelle angegeben, es werden alle aus der Config verwendet")

    # 1. Config laden
    print("\nLade Konfiguration...")
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    all_models = config['models']
    print(f"   {len(all_models)} Modell(e) in config.yaml definiert")

    # 1a. Falls Auswahl angegeben: filtere
    if selected_models:
        models = [m for m in all_models if m['name'] in selected_models]
        unknown = set(selected_models) - {m['name'] for m in all_models}
        if unknown:
            print(f"\nWarnung: folgende Modelle stehen nicht in config.yaml: {', '.join(unknown)}")
        print(f"   {len(models)} werden tatsächlich ausgeführt")
    else:
        models = all_models

    if not models:
        print("\nKeine passenden Modelle gefunden. Breche ab.")
        sys.exit(1)
    
    # 2. Für jedes ausgewählte Modell
    successful = []
    failed = []
    
    for model_spec in models:
        model_name = model_spec['name']
        env_name = model_spec['conda_env']
        script = f"model_scripts/{model_name}.py"
        
        print(f"\n{'='*60}")
        print(f"Modell: {model_name}")
        print(f"   Environment: {env_name}")
        print(f"   Script: {script}")
        print(f"{'='*60}")
        
        try:
            exit_code = run_in_conda_env(env_name, script)
            
            if exit_code == 0:
                print(f"{model_name} erfolgreich")
                successful.append(model_name)
            else:
                print(f"{model_name} fehlgeschlagen")
                failed.append(model_name)
        
        except Exception as e:
            print(f"Fehler bei {model_name}: {e}")
            failed.append(model_name)
    
    # 3. Zusammenfassung
    print(f"\n\n{'='*60}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print(f"Erfolgreich: {len(successful)}")
    for m in successful:
        print(f"   - {m}")
    
    if failed:
        print(f"\nFehlgeschlagen: {len(failed)}")
        for m in failed:
            print(f"   - {m}")
    
    print(f"\n{'='*60}")
    if failed:
        print("Evaluation abgeschlossen (mit Fehlern)")
        sys.exit(1)
    else:
        print("Alle ausgewählten Modelle erfolgreich evaluiert")
        sys.exit(0)

except Exception as e:
    print(f"\nKRITISCHER FEHLER: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
