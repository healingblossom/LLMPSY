import pandas as pd

def load_test_data(filepath):
    """Lädt Test-Daten aus CSV"""
    df = pd.read_csv(filepath)
    return df

def validate_data(data):
    """Prüft, ob Daten valide sind"""
    if data.empty:
        raise ValueError("Daten sind leer!")
    return True