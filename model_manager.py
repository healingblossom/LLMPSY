# model_manager.py Zentrales Skript zum Umgang mit den Modellen (damit sie funktionsfähig sind, und danach (wenn nötig) wieder vom Speicher gelöscht werden)

import os
import yaml
import shutil

from model_scripts.mental_alpaca_wrapper import MentalAlpacaWrapper
#from model_scripts.mistral import MistralWrapper

class ModelManager:
    """Intelligentes Laden und Cleanup von Modellen"""

    # Konstruktor der Modelle, speichert die configs als Class-Parameter und ruft die Pfade auf
    def __init__(self, model_name):
        """Initialisiere mit Config"""
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)

        for each_model in self.config['models']:
            if each_model['name'] == model_name:
                self.model_spec = each_model
        raise ValueError(f"Modell '{model_name}' nicht in config.yaml gefunden!")
    
    def get_model_spec(self):
        """Finde Modell-Spec in Config"""

        return self.model_spec
    
    def load_model(self, model_name):
        """
        Lade ein Modell basierend auf Source-Type.
        Für Server-Modelle: Validiere nur, dass der Pfad existiert.
        """
        model_spec = self.get_model_spec()
        source = model_spec['source']
        
        print(f"\nLade Modell: {model_name}")
        print(f"   Source: {source}")

        if source == 'openrouter':
            return self._load_api_model(model_spec)
        
        elif source == 'local_download':
            return self._load_server_model(model_spec)
        
        elif source == 'hpc':
            return self._load_server_model(model_spec)
        
        else:
            raise NotImplementedError(f"Source '{source}' nicht implementiert")
    
    def _load_server_model(self, model_spec):
        """
        Lade Modell von Server (HPC oder Scratch) mit generate_from_messages()-Methode
        """
        model_name = model_spec['name']
        model_path = model_spec['path']
    
        print(f"  Model Pfad: {model_path}")
    
        # Prüfe, ob Pfad existiert
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modell-Pfad existiert nicht: {model_path}"
                                    f", stell sicher, dass das Modell vorhanden und der richtige Pfad angegeben ist")
    
        print("   Modell-Pfad gefunden, lade Modell...")

        if model_name == "mental_alpaca":
            return MentalAlpacaWrapper()
        elif model_name == "mistral":
            return MistralWrapper(model_path)#todo Wrapper not done yet, look into the MentalAlpacaWrapper for inspo
        else:
            raise ValueError(f"Unbekanntes Modell: {model_name}")
    
    def cleanup_model(self, model_name):
        """
        Cleanup nach Modell-Nutzung.
        Für HPC-Modelle: Nichts löschen!
        """
        model_spec = self.get_model_spec(model_name)
        source = model_spec['source']
        
        print(f"\nCleanup für: {model_name}")
        print(f"   Source: {source}")
        
        if source == 'hpc':
            print("   HPC-Server-Modell wird nicht gelöscht")
        
        elif source == 'local_download':
            model_path = f"{self.scratch_dir}/downloaded_models/{model_name}"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                print(f"   Gelöscht: {model_path}")
            else:
                print(f"   Pfad existiert nicht: {model_path}")


