# utils/data_loader.py
import os
import sys
import pandas as pd
import glob
import openpyxl
import pickle


def convert_patient_excel_to_csv(input_file: str, output_file: str | None = None) -> None:
    """Hilfsfunktion zum Konvertieren einer einzelnen Excel-tabelle zu einer CSV-Tabelle"""
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        if output_file:
            if os.path.exists(output_file):
                confirm = input(f"File '{output_file}' exists. Overwrite? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Operation cancelled.")
                    return
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Converted '{input_file}' to '{output_file}'")
        else:
            df.to_csv(sys.stdout, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error during pandas conversion: {e}", file=sys.stderr)


def convert_all_excels_to_csv(input_dir, output_dir):
    """Konvertieren Ordner an Excel-tabellen zu einem Ordner an CSV-Tabellen,
    Achtung,
    die Interviews haben folgende Namensgebung: Interview_<patienten-id>_<EpisodeTyp>_<zeit>.xlsx
    <EpsisodeTyp> Depression für depressive Episode und Manie für manische/hypomanische Episode steht
    <zeit>: a für aktuell und f für früher steht
    """

    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(input_dir, "Interview_*_*_*.xlsx")
    for xlsx_path in glob.glob(pattern):
        filename = os.path.basename(xlsx_path)
        csv_name = filename.replace(".xlsx", ".csv")
        csv_path = os.path.join(output_dir, csv_name)
        convert_patient_excel_to_csv(xlsx_path, csv_path)


def parse_filename(filename):
    # Erwartetes Muster: "Interview_<id>_<EpisodeTyp>_<zeit>.csv"
    # z.B. "Interview_01_Depression_f.csv"
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    # parts = ["Interview", "01", "Depression", "f"]
    patient_id = parts[1]
    episode_raw = parts[2].lower()
    time_flag = parts[3].lower()  # "f" oder "a"

    # Episode-Typ normalisieren
    if "dep" in episode_raw or "mdd" in episode_raw:
        episode_type = "depression"
    elif "manie" in episode_raw or "mania" in episode_raw:
        episode_type = "mania"
    else:
        raise ValueError(f"Unbekannter Episodentyp in Dateiname: {filename}")

    return patient_id, episode_type, time_flag


def load_all_episodes(csv_dir):
    """

    :param csv_dir: directory mit den csv-tabellen der Patienteninterviews
    :return: dict-datei mit folgendem aufbau: 
    {
      "01": 
      {
        "depression": 
        {
          "f": df_dep_f,   # dataframe-datei für die frühere depressive Episode
          oder
          "a": df_dep_a,   # aktuelle depressive Episode
        },
        "mania": 
        {
          "f": df_mania_f, # dataframe-datei für die frühere manische Episode
          oder
          "a": df_mania_a, # aktuelle manische Episode
        }
      },
      "02": {
        ...
      }
    }
    """
    data = {}
    pattern = os.path.join(csv_dir, "Interview_*_*_*.csv")

    for csv_path in glob.glob(pattern):
        filename = os.path.basename(csv_path)
        patient_id, episode_type, time_flag = parse_filename(filename)
        df = pd.read_csv(csv_path)

        data.setdefault(patient_id, {}).setdefault(episode_type, {})[time_flag] = df

    return data


def get_manic_episodes(data, patient_ids=None):
    """
    getter für manische Episoden aller Patienten
    :param data: siehe return load_all_episodes
    :param patient_ids: es kann eine Liste im Format ["id1","id2","id3","id4"] gegeben werden, um nur diese Patiententranskripte auszugeben
    """
    result = {}

    if patient_ids is not None:
        patient_ids = set(str(pid) for pid in patient_ids)

    for patient_id, episodes in data.items():
        if patient_ids is not None and str(patient_id) in patient_ids:
            mania_dict = episodes.get("mania", {})
            if not mania_dict:
                continue
            else:
                result[patient_id] = mania_dict
    return result


def get_depressive_episodes(data, patient_ids=None):
    """
    getter für depressive Episoden aller Patienten
    :param data: siehe return load_all_episodes
    :param patient_ids: es kann eine Liste im Format ["id1","id2","id3","id4"] gegeben werden, um nur diese Patiententranskripte auszugeben
    """
    result = {}

    if patient_ids is not None:
        patient_ids = set(str(pid) for pid in patient_ids)

    for patient_id, episodes in data.items():
        if patient_ids is not None and str(patient_id) in patient_ids:
            depression_dict = episodes.get("depression", {})
            if not depression_dict:
                continue
            else:
                result[patient_id] = depression_dict
    return result


def get_full_transcripts(data, time_order=("a", "f"), patient_ids=None):
    """
    Gibt pro Patient ein DataFrame zurück, in dem Episoden in einer konfigurierbaren Reihenfolge
    hintereinandergehängt sind. Es sortiert dabei immer nach Zeit, gibt aber abhängig von der
    Reihenfolge in time_order erst frühere oder erst aktuelle Episoden an, sortiert Episoden intern
    aber erst nach depression, dann manie

    :param data: siehe return load_all_episodes
    :param time_order: konfigurierbarer String-Tupel, in welcher Reihenfolge aktuelle und frühere Episoden herauszugeben werden
    :param patient_ids: es kann eine Liste im Format ["id1","id2","id3","id4"] gegeben werden, um nur diese Patiententranskripte auszugeben
    """
    result = {}
    episode_order = ("depression", "mania")

    if patient_ids is not None:
        patient_ids = set(str(pid) for pid in patient_ids)

    for patient_id, episodes in data.items():
        if patient_ids is not None and str(patient_id) not in patient_ids:
            continue

        dfs = []

        for t in time_order:
            for e in episode_order:
                df = episodes.get(e, {}).get(t)
                if df is not None:
                    dfs.append(df.assign(episode=e, time=t))

        if dfs:
            result[patient_id] = pd.concat(dfs, ignore_index=True)

    return result


if __name__ == '__main__':
    # filepath = "./Interview_01_Full.csv"
    # convert_patient_excel_to_csv("./excel_data/Interview_01_Manie_a.xlsx", "./csv_data/Interview_01_Manie_a.csv")
    # convert_all_excels_to_csv("./excel_data/", "./csv_data/")

    dictionary = load_all_episodes("./csv_data/")
    manische_episoden = get_manic_episodes(dictionary, patient_ids=["01", "05", "08"])
    depressive_episoden = get_depressive_episodes(dictionary, patient_ids=["01", "05", "08"])
    full_transcripts = get_full_transcripts(dictionary, time_order=("f", "a"), patient_ids=["01"])

    print("_________________________dictionary_____________________")
    # print(dictionary)
    print("_________________________manische_episoden_____________________")
    print(manische_episoden)
    print("_________________________depressive_episoden_____________________")
    print(depressive_episoden)
    print("_________________________full_transcripts_____________________")
    print(full_transcripts)


