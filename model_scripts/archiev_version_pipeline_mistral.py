# Skript zur Bearbeitung aller tasks durch mistral, 
# läuft in der gpuland_env Enviroment

import yaml
import json
import sys
import os

# Damit die model_manager.py gefunden wird
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# path.abspath konvertiert den (relativen) file-path zu einem sicher absoluten path, path.dirname nimmt den vorherigen directory-name raus und so bewegen wir uns ins LLM-PSY directory, von dem aus man model_manager finden kann.

from model_manager import ModelManager

print("\n" + "="*60)
print("PIPELINE: mistral")
print("="*60)

try:
    # 1. Config laden
    print("\n1)  Lade Config...")
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    print("   Config geladen")
    
    # 2. ModelManager erstellen
    print("\n2)  Erstelle ModelManager...")
    manager = ModelManager('config.yaml')
    
    # 3. Modell laden
    print("\n3)  Lade Modell...")
    model = manager.load_model('mistral')
    print(f"   Modell geladen: {model}")
    
    # 4. Daten laden (TODO: sobald du die richtigen daten hast, die du laden könntest, ladest du sie erst hier, dann später mit utils.data_loader)
    print("\n4)  Lade Test-Daten...")
    input_file = f"{config['paths']['scratch_directory']}/data/dummy.csv"
    print(f"   Pfad: {input_file}")
    
    # Prüfe, ob Datei existiert (einfache Validierung)
    if os.path.exists(input_file):
        print("   Daten-Datei existiert")
        # TODO: Später echtes Laden
        test_data = {'samples': 100, 'features': 20} # erstellen ein directory mit einem eintrag
    else:
        print("   Daten-Datei nicht gefunden (wird später erstellt)")
        test_data = {'samples': 0, 'features': 0} 
    
    # 5. Dummy-Evaluierung, wird raus genommen, sobald wie möglich
    print("\n5)  Evaluiere Modell...")
    results = {
        'model_name': 'custom_model',
        'task': 'bipolar_diagnosis',
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.89,
        'f1_score': 0.87,
        'test_samples': test_data['samples'],
        'status': 'success'
    }
    print("   Evaluierung abgeschlossen")
    print(f"   Ergebnisse: {results}")

    # 5.5. Modell-Ausgabe sauber machen, so wie nötig
    
    # 6. Speichere Ergebnisse (auf SAFE Location!), wird überarbeitet, wie nötig
    print("\n6)  Speichere Ergebnisse...")
    output_dir = config['paths']['proj_directory'] + '/results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/mistral_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Ergebnisse gespeichert: {output_file}")
    
    # 7. Cleanup
    print("\n7)  Cleanup...")
    manager.cleanup_model('mistral')
    
    print("\n" + "="*60)
    print("custom_model FERTIG")
    print("="*60 + "\n")

except Exception as e:
    print("\nFEHLER:")
    print(e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
