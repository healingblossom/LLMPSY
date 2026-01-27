import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from pathlib import Path
import json


class InterviewDataParser:
    """
    Parser für Excel-Dateien mit Patienteninterviews und Expert-Ground-Truth.
    Konvertiert unstrukturierte Excel-Daten in eine strukturierte Python-Datenstruktur.
    """

    def __init__(self, excel_file_path: str, verbose: bool = False):
        """
        Initialisiert den Parser mit einer Excel-Datei.

        Args:
            excel_file_path: Pfad zur Excel-Datei
            verbose: Wenn True, gibt detaillierte Debug-Informationen aus
        """
        self.excel_file = excel_file_path
        self.data_structure = {}
        self.verbose = verbose

    def _log(self, message: str):
        """Hilfsmethod für Logging"""
        if self.verbose:
            #print(f"[DEBUG] {message}")
            pass

    # ==================== PARSING METHODEN ====================

    def load_interview_sheets(self) -> Dict[str, pd.DataFrame]:
        """
        Lädt alle Interview-Tabellen aus der Excel-Datei.

        Returns:
            Dictionary mit Sheet-Namen als Keys und DataFrames als Values
        """
        xl_file = pd.ExcelFile(self.excel_file)
        interview_sheets = {}

        for sheet_name in xl_file.sheet_names:
            if sheet_name.lower().startswith('interview_'):
                df = pd.read_excel(self.excel_file, sheet_name=sheet_name, header=None)
                interview_sheets[sheet_name] = df
                self._log(f"Geladen: {sheet_name}, Shape: {df.shape}")

        return interview_sheets

    def extract_patient_id_and_language(self, sheet_name: str) -> tuple[str | None, str | None]:
        """
        Extrahiert die Patienten-ID und die Sprache aus dem Sheet-Namen.
        Erwartetes Format: 'Interview_01_de' oder 'Interview_01_en'.
        Rückgabe: (patient_id, sprache), z.B. ('01', 'de').
        Falls kein Match: (None, None).
        """
        match = re.search(r'interview_(\d+)_(de|en)', sheet_name, re.IGNORECASE)
        if match:
            patient_id = match.group(1)
            language = match.group(2).lower()
            return patient_id, language
        return None, None

    def _is_episode_header(self, cell_value) -> Tuple[bool, Optional[str]]:
        """
        Prüft, ob eine Zelle ein Episode-Header ist und gibt den Typ zurück.
        """
        if pd.isna(cell_value):
            return False, None

        cell_str = str(cell_value).lower().strip()

        # Überprüfe auf verschiedene Varianten
        if 'depression' in cell_str or 'depressive' in cell_str:
            if 'aktuelle' in cell_str or 'current' in cell_str:
                return True, 'aktuelle_depression'
            elif 'früher' in cell_str or 'past' in cell_str or 'previous' in cell_str:
                return True, 'frühere_depression'

        if 'manie' in cell_str or 'hypomanie' in cell_str or 'mania' in cell_str or 'manisch' in cell_str:
            if 'aktuelle' in cell_str or 'current' in cell_str:
                return True, 'aktuelle_mania'
            elif 'früher' in cell_str or 'past' in cell_str or 'previous' in cell_str:
                return True, 'frühere_mania'

        return False, None

    def _extract_symptom_list(self, cell_value) -> List[str]:
        """
        Konvertiert eine Symptom-Zelle (String oder Liste) in eine Liste von Strings.
        """
        if pd.isna(cell_value):
            return []

        cell_str = str(cell_value).strip()

        if not cell_str:
            return []

        # Versuche verschiedene Formate zu parsen
        if cell_str.startswith('[') and cell_str.endswith(']'):
            try:
                return eval(cell_str)
            except:
                pass

        # Fallback: Spalte durch Komma oder Semikolon
        if ',' in cell_str:
            return [s.strip() for s in cell_str.split(',') if s.strip()]
        elif ';' in cell_str:
            return [s.strip() for s in cell_str.split(';') if s.strip()]

        return [cell_str] if cell_str else []

    def _is_criteria_row(self, cell_value) -> bool:
        """
        Prüft ob eine Zeile in Spalte F Kriterium-Daten enthält (nicht leer/NaN).

        Die Excel-Struktur: Nach jedem Transkript folgen 3 Kriterium-Zeilen:
        - Zeile 1: Spalte F enthält erfüllte Kriterien
        - Zeile 2: Spalte F enthält nicht erfüllte Kriterien
        - Zeile 3: Spalte F enthält unbekannte/fehlende Kriterien

        Returns:
            True wenn Spalte F gefüllt ist (es ist eine Kriterium-Zeile),
            False wenn Spalte F leer ist
        """
        return pd.notna(cell_value) and str(cell_value).strip() != ''

    def _extract_criteria_values(self, cell_value) -> List[str]:
        """
        Extrahiert die Kriterium-Werte direkt aus einer Kriterium-Zeile in Spalte F.

        Die Zellenwerte können verschiedene Formate haben:
        - Liste: ['Kriterium A', 'Kriterium B']
        - Komma-getrennt: 'Kriterium A, Kriterium B'
        - Semikolon-getrennt: 'Kriterium A; Kriterium B'

        Args:
            cell_value: Wert aus Spalte F (eine Zeile einer 3er-Gruppe)

        Returns:
            List von Kriterium-Strings
        """
        if pd.isna(cell_value):
            return []

        cell_str = str(cell_value).strip()

        if not cell_str:
            return []

        # Nutze bestehende Symptom-Parser-Logik
        return self._extract_symptom_list(cell_str)

    def _parse_episode_section(self, df: pd.DataFrame, start_row: int) -> Tuple[pd.DataFrame, Dict, int]:
        """
        Parst eine komplette Episode-Sektion (Daten + diagnostische Kriterien).

        Returns:
            Tuple (episode_dataframe, diagnostic_criteria_dict, next_row)
        """
        episode_data = []
        diagnostic_criteria = {
            'erfüllte Kriterien': [],
            'nicht erfüllte Kriterien': [],
            'unbekannte Kriterien': []
        }

        current_row = start_row

        self._log(f"  Starte Episoden-Parsing bei Zeile {current_row}")

        # Phase 1: Parse Episode-Daten bis zu Kriterien
        while current_row < len(df):
            # Prüfe auf nächsten Episode-Header
            col_a = df.iloc[current_row, 0] if len(df.columns) > 0 else None
            is_header, _ = self._is_episode_header(col_a)

            if is_header and current_row > start_row:
                self._log(f"    Nächster Episode-Header gefunden bei Zeile {current_row}")
                break

            # Prüfe auf Kriterium-Label in Spalte F
            col_f = df.iloc[current_row, 5] if len(df.columns) > 5 else None
            has_criteria = self._is_criteria_row(col_f)

            if has_criteria:
                self._log(f"    Kriterium-Sequenz gefunden ab Zeile {current_row}")

                # Parse die 3 aufeinanderfolgenden Kriterium-Zeilen
                for i in range(3):
                    if current_row + i < len(df):
                        col_f_value = df.iloc[current_row + i, 5] if len(df.columns) > 5 else None
                        values = self._extract_criteria_values(col_f_value)

                        # Bestimme den Typ basierend auf Position in der Sequenz
                        if i == 0:
                            diagnostic_criteria['erfüllte Kriterien'] = values
                            self._log(f"      [Zeile {current_row}] Erfüllte Kriterien: {values}")
                        elif i == 1:
                            diagnostic_criteria['nicht erfüllte Kriterien'] = values
                            self._log(f"      [Zeile {current_row + 1}] Nicht erfüllte Kriterien: {values}")
                        elif i == 2:
                            diagnostic_criteria['unbekannte Kriterien'] = values
                            self._log(f"      [Zeile {current_row + 2}] Unbekannte Kriterien: {values}")

                current_row += 3
                break

            # Normale Datenzeile
            col_b = df.iloc[current_row, 1] if len(df.columns) > 1 else None  # Chunk
            col_c = df.iloc[current_row, 2] if len(df.columns) > 2 else None  # Transkript
            col_d = df.iloc[current_row, 3] if len(df.columns) > 3 else None  # Relevant Section
            col_e = df.iloc[current_row, 4] if len(df.columns) > 4 else None  # Symptome

            # Überspringe leere Zeilen, aber nicht wenn nur col_a gefüllt ist (wegen merged cells)
            if pd.notna(col_b) or pd.notna(col_c) or pd.notna(col_d) or pd.notna(col_e):
                episode_data.append({
                    'chunk': col_b,
                    'transcript': col_c,
                    'relevant_section': col_d,
                    'symptoms': self._extract_symptom_list(col_e)
                })
                self._log(f"    Chunk-Zeile hinzugefügt: {col_b}")

            current_row += 1

        # Konvertiere zu DataFrame
        if episode_data:
            episode_df = pd.DataFrame(episode_data)
        else:
            episode_df = pd.DataFrame(columns=['chunk', 'transcript', 'relevant_section', 'symptoms'])

        return episode_df, diagnostic_criteria, current_row

    def _parse_overall_diagnostic_lines(self, df: pd.DataFrame, row_idx: int) -> Dict[str, any]:
        """
        Parst die Zusammenfassung am Ende einer Interview-Tabelle.
        """
        summary = {
            'Verdachtsdiagnose': None,
            'Notwendige Differentialdiagnosen': None,
            'Komorbiditäten': None,
            'Komplexität des Interviews': None,
            'Sicherheit der Verdachtsdiagnose': None
        }

        if row_idx >= len(df):
            return summary

        row = df.iloc[row_idx]

        # Spalte G, H, I, J, K (Index 6, 7, 8, 9, 10)
        col_g = row[6] if len(row) > 6 else None
        col_h = row[7] if len(row) > 7 else None
        col_i = row[8] if len(row) > 8 else None
        col_j = row[9] if len(row) > 9 else None
        col_k = row[10] if len(row) > 10 else None

        # Prüfe ob dies eine Header-Zeile ist
        if pd.notna(col_g) and str(col_g).lower().strip() not in ['verdachtsdiagnose', 'verdacht', 'diagnosis']:
            summary['Verdachtsdiagnose'] = str(col_g).strip()

        if pd.notna(col_h) and str(col_h).lower().strip() not in ['notwendige differentialdiagnose',
                                                                  'differentialdiagnose']:
            summary['Notwendige Differentialdiagnosen'] = str(col_h).strip()

        if pd.notna(col_i) and str(col_i).lower().strip() not in ['komorbiditäten', 'comorbidities']:
            summary['Komorbiditäten'] = str(col_i).strip()

        if pd.notna(col_j):
            try:
                val = int(col_j) if not pd.isna(col_j) else None
                if val is not None and str(col_j).lower().strip() not in ['komplexität', 'complexity']:
                    summary['Komplexität des Interviews'] = val
            except:
                pass

        if pd.notna(col_k):
            try:
                val = int(col_k) if not pd.isna(col_k) else None
                if val is not None and str(col_k).lower().strip() not in ['sicherheit', 'certainty']:
                    summary['Sicherheit der Verdachtsdiagnose'] = val
            except:
                pass

        return summary

    def _find_overall_diagnostic_lines_row(self, df: pd.DataFrame, start_from: int = 0) -> int:
        """
        Findet die Zeile mit der Zusammenfassung (Verdachtsdiagnose, etc.).
        """
        for row_idx in range(start_from, len(df)):
            col_g = df.iloc[row_idx, 6] if len(df.columns) > 6 else None

            if pd.notna(col_g):
                col_g_str = str(col_g).lower().strip()
                # Prüfe ob das eine echte Diagnose ist, nicht der Header
                if 'verdacht' in col_g_str and col_g_str not in ['verdachtsdiagnose', 'verdacht']:
                    return row_idx

        return -1

    def parse_interview_sheet(self, df: pd.DataFrame, patient_id: str, language: str) -> Dict:
        """
        Parst eine komplette Interview-Tabelle.
        """
        self._log(f"\n=== Parse Patient {patient_id} ===")
        self._log(f"DataFrame Shape: {df.shape}")

        if language not in ['de', 'en']:
            raise ValueError(f"Sprache {language} unbekannt.")

        patient_data = {
            'language': {language},
            'interviews': {
                'depression': {},
                'mania': {}
            },
            'insgesammt': {}
        }

        current_row = 0
        episode_mapping = {
            'aktuelle_depression': ('depression', 'aktuelle'),
            'frühere_depression': ('depression', 'frühere'),
            'aktuelle_mania': ('mania', 'aktuelle'),
            'frühere_mania': ('mania', 'frühere')
        }

        # Finde zuerst die Summary-Zeile um zu wissen wo sie ist
        summary_row = self._find_overall_diagnostic_lines_row(df)
        self._log(f"Summary-Zeile gefunden bei: {summary_row}")

        while current_row < len(df):
            # Prüfe auf Episode-Header
            col_a = df.iloc[current_row, 0] if len(df.columns) > 0 else None
            is_header, episode_type = self._is_episode_header(col_a)

            if is_header and episode_type:
                self._log(f"Zeile {current_row}: Episode-Header gefunden: {episode_type}")
                current_row += 0  # Springe über Header #todo schau ob es jetzt funtkioniert

                # Parse die Episode
                episode_df, criteria, current_row = self._parse_episode_section(df, current_row)

                # Speichere die Daten
                if episode_type in episode_mapping:
                    disorder_type, time_frame = episode_mapping[episode_type]
                    patient_data['interviews'][disorder_type][time_frame] = episode_df
                    if 'Diagnostische Kriterien' not in patient_data['interviews'][disorder_type]:
                        patient_data['interviews'][disorder_type]['Diagnostische Kriterien'] = {}
                    patient_data['interviews'][disorder_type]['Diagnostische Kriterien'][time_frame] = criteria
                    self._log(f"  Gespeichert: {disorder_type}/{time_frame}")
            else:
                # Prüfe ob wir bei der Summary-Zeile sind
                if summary_row != -1 and current_row == summary_row:
                    self._log(f"Zeile {current_row}: Summary-Zeile gefunden")
                    patient_data['insgesammt'] = self._parse_overall_diagnostic_lines(df, current_row)
                    self._log(f"  Summary geparst: {patient_data['insgesammt']}")
                    break

                current_row += 1

        return patient_data

    def parse_all_interviews(self) -> Dict[str, Dict]:
        """
        Parst alle Interview-Tabellen aus der Excel-Datei. Erstellt die gewünschte Datenstruktur zum Bearbeiten.
        """
        interview_sheets = self.load_interview_sheets()

        for sheet_name, df in interview_sheets.items():
            patient_id, language = self.extract_patient_id_and_language(sheet_name)
            patient_data = self.parse_interview_sheet(df, patient_id, language)
            self.data_structure[patient_id] = patient_data

        return self.data_structure

    # ==================== HILFSMETHODEN FÜR EXPORT ====================

    def export_symptoms_to_json(self, output_path: str, episode_type: Optional[str] = None):
        """
        Exportiert Symptome im geforderten Format zu JSON.

        Args:
            output_path: Pfad zur Ausgabe-Datei
            episode_type: Optional - 'depression' oder 'mania'

        Beispiel:
            parser.export_symptoms_to_json('symptoms.json', episode_type='depression')
        """
        symptoms = self.get_symptoms(episode_type=episode_type)

        # Konvertiere zu JSON-kompatiblem Format
        json_data = {}
        for patient_id, disorders in symptoms.items():
            json_data[patient_id] = {}
            for disorder, times in disorders.items():
                json_data[patient_id] = {}
                for time, chunks in times.items():
                    json_data[patient_id][time] = chunks

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Symptome exportiert zu: {output_path}")


    def export_diagnostic_criteria_to_json(self, output_path: str, episode_type: Optional[str] = None):
        """
        Exportiert diagnostische Kriterien zu JSON.

        Args:
            output_path: Pfad zur Ausgabe-Datei
            episode_type: Optional - 'depression' oder 'mania'

        Beispiel:
            parser.export_diagnostic_criteria_to_json('criteria.json')
        """
        criteria = self.get_diagnostic_criteria(episode_type=episode_type)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(criteria, f, indent=2, ensure_ascii=False)

        print(f"Diagnostische Kriterien exportiert zu: {output_path}")


    def export_summary_to_csv(self, output_path: str, groups: Optional[List[str]] = None, patient_ids: Optional[List[str]] = None):
        """
        Exportiert Summary-Daten zu CSV.

        Args:
            output_path: Pfad zur Ausgabe-Datei
            groups: Optional - Liste der Gruppen
            patient_ids: Optional - Liste der Patient-IDs

        Beispiel:
            parser.export_summary_to_csv('summary.csv')
        """
        summary = self.get_summary_data(groups=groups, patient_ids=patient_ids)

        # Konvertiere zu DataFrame
        df = pd.DataFrame(summary)
        df.to_csv(output_path, encoding='utf-8', index_label='Patient ID')

        print(f"Summary exportiert zu: {output_path}")

    # =======================================================================
    #                   METHODEN VERWENDUNG AUẞERHALB DIESER KLASSE
    # =======================================================================

    # ==================== DISPLAY & EXPORT METHODEN ====================

    def print_patient_transcript(self, patient_id: str, episode_type: str = None, time_frame: str = None):
        """
        Gibt das komplette Transkript für einen Patienten oder eine Episode aus.

        Args:
            patient_id: Patient-ID (z.B. '01', achtet auf führende Nullen)
            episode_type: 'depression' oder 'mania' (optional)
            time_frame: 'aktuelle' oder 'frühere' (optional)
        """


        if patient_id not in self.data_structure:
            print(f"Patient {patient_id} nicht gefunden.")
            return

        patient_data = self.data_structure[patient_id]

        print(f"PATIENT {patient_id}")
        print(f"{'=' * 80}")

        # Zeige Zusammenfassung
        summary = patient_data['insgesammt']
        print(f"\n ZUSAMMENFASSUNG:")
        print(f"  Verdachtsdiagnose: {summary.get('Verdachtsdiagnose')}")
        print(f"  Differentialdiagnosen: {summary.get('Notwendige Differentialdiagnosen')}")
        print(f"  Komorbiditäten: {summary.get('Komorbiditäten')}")
        print(f"  Komplexität: {summary.get('Komplexität des Interviews')}/10")
        print(f"  Sicherheit: {summary.get('Sicherheit der Verdachtsdiagnose')}/10")

        # Bestimme welche Episodes zu zeigen sind
        episodes_to_show = []

        for disorder in ['depression', 'mania']:
            if episode_type and episode_type != disorder:
                continue

            disorder_data = patient_data['interviews'].get(disorder, {})

            for time in ['aktuelle', 'frühere']:
                if time_frame and time_frame != time:
                    continue

                if time in disorder_data:
                    episodes_to_show.append((disorder, time, disorder_data[time],
                                             disorder_data.get('Diagnostische Kriterien', {}).get(time, {})))

        # Zeige jede Episode
        for disorder, time, episode_df, criteria in episodes_to_show:
            disorder_name = "Depression" if disorder == "depression" else "Manie/Hypomanie"
            time_name = "Aktuelle" if time == "aktuelle" else "Frühere"

            print(f" {time_name} {disorder_name.upper()}")
            print(f"{'─' * 80}")

            # Zeige diagnostische Kriterien
            print(f"\n Erfüllte Kriterien:")
            for criterion in criteria.get('erfüllte Kriterien', []):
                print(f"    • {criterion}")

            print(f"\n Nicht erfüllte Kriterien:")
            for criterion in criteria.get('nicht erfüllte Kriterien', []):
                print(f"    • {criterion}")

            print(f"\n Unbekannte Kriterien:")
            for criterion in criteria.get('unbekannte Kriterien', []):
                print(f"    • {criterion}")

            # Zeige Transkript
            print(f"\n TRANSKRIPT ({len(episode_df)} Chunks):")
            print(f"  {'-' * 76}")

            for idx, row in episode_df.iterrows():
                chunk = row['chunk']
                transcript = row['transcript']
                relevant = row['relevant_section']
                symptoms = row['symptoms']

                print(f"\n    [{chunk}]")

                if pd.notna(transcript) and str(transcript).strip():
                    transcript_text = str(transcript)
                    if len(transcript_text) > 75:
                        wrapped = "\n    ".join([transcript_text[i:i + 75] for i in range(0, len(transcript_text), 75)])
                        print(f"    {wrapped}")
                    else:
                        print(f"    {transcript_text}")

                if pd.notna(relevant) and str(relevant).strip():
                    print(f" Relevant: {str(relevant)}")

                if symptoms and len(symptoms) > 0:
                    print(f" Symptome: {', '.join(symptoms)}")


    def print_patient_structure(self, patient_id: str):
        """
        Gibt die Struktur eines Patienten in übersichtlicher Form aus.
        """
        if patient_id not in self.data_structure:
            print(f"Patient {patient_id} nicht gefunden.")
            return

        patient_data = self.data_structure[patient_id]

        print(f"STRUKTUR PATIENT {patient_id}")

        # Zusammenfassung
        print(f"\n📋 insgesammt:")
        for key, value in patient_data['insgesammt'].items():
            print(f"  {key}: {value}")

        # Interviews
        print(f"\n📝 interviews:")
        for disorder in ['depression', 'mania']:
            disorder_data = patient_data['interviews'].get(disorder, {})

            print(f"\n  {disorder.upper()}:")

            for time in ['aktuelle', 'frühere']:
                if time in disorder_data:
                    df = disorder_data[time]
                    if isinstance(df, pd.DataFrame):
                        print(f"    {time}: {len(df)} Chunks")
                        print(f"      Spalten: {list(df.columns)}")
                    else:
                        print(f"    {time}: {type(df)}")

            # Diagnostische Kriterien
            if 'Diagnostische Kriterien' in disorder_data:
                criteria = disorder_data['Diagnostische Kriterien']
                print(f"    Diagnostische Kriterien:")
                for time, crit_dict in criteria.items():
                    print(f"      {time}:")
                    for crit_type, crit_list in crit_dict.items():
                        print(f"        {crit_type}: {len(crit_list)} items")

    def export_patient_to_json(self, patient_id: str, output_path: str = None) -> Dict:
        """
        Exportiert Patienten-Daten als JSON.
        """
        if patient_id not in self.data_structure:
            print(f"Patient {patient_id} nicht gefunden.")
            return None

        patient_data = self.data_structure[patient_id]

        # Konvertiere DataFrames zu Dicts
        export_data = {
            'patient_id': patient_id,
            'insgesammt': patient_data['insgesammt'],
            'interviews': {
                'depression': {},
                'mania': {}
            }
        }

        for disorder in ['depression', 'mania']:
            disorder_data = patient_data['interviews'].get(disorder, {})

            for time in ['aktuelle', 'frühere']:
                if time in disorder_data and isinstance(disorder_data[time], pd.DataFrame):
                    export_data['interviews'][time] = disorder_data[time].to_dict('records')

            if 'Diagnostische Kriterien' in disorder_data:
                export_data['interviews']['Diagnostische Kriterien'] = disorder_data[
                    'Diagnostische Kriterien']

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"Export gespeichert: {output_path}")

        return export_data


    # ==================== GETTER METHODEN ====================

    def get_data_structure(self) -> Dict[str, Dict]:
        """
        Gibt die geparste Datenstruktur zurück.
        """
        return self.data_structure

    def get_transcripts(self, episode_type: Optional[str] = None, include_time_keys: bool = True) -> Dict[str, Dict]:
        """
        Gibt alle Patienten-Transkripte gruppiert nach:
        Patient → Episodentyp (depression/mania) → Einzelne Episoden (aktuelle/frühere)

        Args:
            episode_type: Optional - 'depression' oder 'mania' zum Filtern
                         Wenn None, werden alle zurückgegeben

        Returns:
            {
                'patient_01': {
                    'depression': {
                        'aktuelle': pd.DataFrame,
                        'frühere': pd.DataFrame
                    },
                    'mania': {
                        'aktuelle': pd.DataFrame,
                        'frühere': pd.DataFrame
                    }
                },
                ...
            }

        Beispiel:
            # Alle Transkripte
            all_transcripts = parser.get_all_transcripts()

            # Nur Depression
            dep_transcripts = parser.get_all_transcripts(episode_type='depression')
        """
        if episode_type not in ['depression', 'mania', None]:
            raise ValueError(f"episode_type muss 'depression', 'mania' oder None sein, nicht '{episode_type}'")

        result = {}

        for patient_id, patient_data in self.data_structure.items():
            result[patient_id] = {}

            for disorder in ['depression', 'mania']:
                # Filter nach episode_type wenn gesetzt
                if episode_type and episode_type != disorder:
                    continue

                disorder_data = patient_data['interviews'].get(disorder, {})

                if not disorder_data:
                    continue

                if include_time_keys:
                    # Mit Zeit-Keys: 'aktuelle' / 'frühere'
                    result[patient_id][disorder] = {}

                    for time in ['aktuelle', 'frühere']:
                        if time in disorder_data:
                            df = disorder_data[time].copy()

                            # Entferne symptoms-Spalte wenn gewünscht
                            df = df.drop(columns=['relevant_section'])
                            df = df.drop(columns=['symptoms'])
                            df = df.drop(columns=['chunk'])

                            result[patient_id][disorder][time] = df

                else:
                    # Ohne Zeit-Keys: DataFrame kombinieren
                    dfs_to_concat = []

                    for time in ['aktuelle', 'frühere']:
                        if time in disorder_data:
                            df = disorder_data[time].copy()

                            # Entferne symptoms-Spalte wenn gewünscht
                            df = df.drop(columns=['relevant_section'])
                            df = df.drop(columns=['symptoms'])
                            df = df.drop(columns=['chunk'])

                            dfs_to_concat.append(df)

                    if dfs_to_concat:
                        combined_df = pd.concat(dfs_to_concat, ignore_index=True)
                        result[patient_id][disorder] = combined_df

        return result


    def get_symptoms(self, episode_type: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Dict[str, List[Dict]]]]]:
        """
        Gibt alle Symptome für alle Patienten mit markierten Textstellen.

        Args:
            episode_type: Optional - 'depression' oder 'mania' zum Filtern

        Returns:
            {
                'patient_01': {
                    'depression': {
                        'aktuelle': {
                            'chunk_1': [
                                {'symptom': 'Schlaflosigkeit', 'section': 'Ich kann nachts nicht schlafen...'},
                                {'symptom': 'Traurigkeit', 'section': 'Ich fühle mich sehr traurig...'}
                            ],
                            'chunk_2': [...]
                        },
                        'frühere': {...}
                    },
                    'mania': {...}
                },
                ...
            }

        Beispiel:
            all_symptoms = parser.get_all_symptoms()

            # Nur Depression-Symptome
            dep_symptoms = parser.get_all_symptoms(episode_type='depression')
        """
        if episode_type not in ['depression', 'mania', None]:
            raise ValueError(f"episode_type muss 'depression' oder 'mania' sein, nicht '{episode_type}'")

        result = {}

        for patient_id, patient_data in self.data_structure.items():
            result[patient_id] = {}

            for disorder in ['depression', 'mania']:
                if episode_type and episode_type != disorder:
                    continue

                result[patient_id] = {}
                disorder_data = patient_data['interviews'].get(disorder, {})

                for time in ['aktuelle', 'frühere']:
                    if time in disorder_data:
                        episode_df = disorder_data[time]
                        result[patient_id][disorder][time] = {}

                        if isinstance(episode_df, pd.DataFrame) and len(episode_df) > 0:
                            for idx, row in episode_df.iterrows():
                                chunk = row['chunk']
                                symptoms = row['symptoms']
                                relevant_section = row['relevant_section']

                                if chunk not in result[patient_id][time]:
                                    result[patient_id][disorder][time][chunk] = []

                                # Erstelle Symptom-Einträge im geforderten Format
                                if symptoms and len(symptoms) > 0:
                                    for symptom in symptoms:
                                        result[patient_id][disorder][time][chunk].append({
                                            'symptom': symptom,
                                            'section': str(relevant_section) if pd.notna(relevant_section) else str(
                                                row['transcript'])
                                        })

        return result

    def get_diagnostic_criteria(self) -> Dict[str, Dict]:
        """
        Gibt alle diagnostischen Kriterien für alle Patienten zurück.

        Returns:
            {
                'patient_01': {
                    'depression': {
                        'aktuelle': {
                            'erfüllte Kriterien': ['Kriterium 1', 'Kriterium 2'],
                            'nicht erfüllte Kriterien': ['Kriterium 3'],
                            'unbekannte Kriterien': []
                        },
                        'frühere': {...}
                    },
                    'mania': {...}
                },
                ...
            }

        Beispiel:
            all_criteria = parser.get_all_diagnostic_criteria()

            # Nur Depression
            dep_criteria = parser.get_all_diagnostic_criteria(episode_type='depression')
        """
        result = {}

        for patient_id, patient_data in self.data_structure.items():
            result[patient_id] = {}

            for disorder in ['depression', 'mania']:
                result[patient_id] = {}
                disorder_data = patient_data['interviews'].get(disorder, {})
                criteria_data = disorder_data.get('Diagnostische Kriterien', {})

                times = ['aktuelle', 'frühere']

                for time in times:
                    if time in criteria_data:
                        result[patient_id][disorder][time] = criteria_data[time]

        return result

    def get_summary_data(self, groups: Optional[List[str]] = None, patient_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Gibt die Zusammenfassungs-Inhalte gruppiert nach Spalten zurück.

        Args:
            groups: Liste der gewünschten Gruppen/Spalten. Wenn None, alle zurückgeben.
                   Mögliche Werte:
                   - 'Verdachtsdiagnose'
                   - 'Notwendige Differentialdiagnosen'
                   - 'Komorbiditäten'
                   - 'Komplexität des Interviews'
                   - 'Sicherheit der Verdachtsdiagnose'

            patient_ids: Liste von Patient-IDs zum Filtern. Wenn None, alle zurückgeben.

        Returns:
            {
                'Verdachtsdiagnose': {
                    'patient_01': 'Bipolare Störung Typ I',
                    'patient_02': 'Bipolare Störung Typ II',
                    ...
                },
                'Notwendige Differentialdiagnosen': {
                    'patient_01': 'Depression, Angststörung',
                    ...
                },
                ...
            }

        Beispiel:
            # Alle Summary-Daten
            all_summary = parser.get_summary_data()

            # Nur bestimmte Spalten
            diagnoses = parser.get_summary_data(groups=['Verdachtsdiagnose', 'Notwendige Differentialdiagnosen'])

            # Nur bestimmte Patienten
            selected = parser.get_summary_data(patient_ids=['01', '02', '03'])

            # Kombination
            filtered = parser.get_summary_data(
                groups=['Verdachtsdiagnose', 'Komplexität des Interviews'],
                patient_ids=['01', '02']
            )
        """
        # Definiere alle möglichen Gruppen
        all_groups = [
            'Verdachtsdiagnose',
            'Notwendige Differentialdiagnosen',
            'Komorbiditäten',
            'Komplexität des Interviews',
            'Sicherheit der Verdachtsdiagnose'
        ]

        # Bestimme welche Gruppen zu verwenden sind
        groups_to_use = groups if groups is not None else all_groups

        # Validiere die Gruppen
        invalid_groups = [g for g in groups_to_use if g not in all_groups]
        if invalid_groups:
            raise ValueError(f"Ungültige Gruppen: {invalid_groups}. Gültig: {all_groups}")

        # Bestimme, welche Patienten zu verwenden sind
        patients_to_use = patient_ids if patient_ids is not None else list(self.data_structure.keys())

        # Filtere Patienten die nicht existieren
        patients_to_use = [p for p in patients_to_use if p in self.data_structure]

        result = {}

        # Initialisiere die Gruppen
        for group in groups_to_use:
            result[group] = {}

        # Fülle die Daten
        for patient_id in patients_to_use:
            patient_data = self.data_structure[patient_id]
            summary = patient_data.get('insgesammt', {})

            for group in groups_to_use:
                result[group][patient_id] = summary.get(group)

        return result



"""
# Beispiel-Nutzung:
if __name__ == "__main__":
    # Initialisiere Parser mit verbose mode
    parser = InterviewDataParser('/home/blossom/PycharmProjects/LLMPSY/data/LLM-PSY_Labeln_Synthetische_Daten_v1.0.xlsm', verbose=True)


    # Optional: Inspiziere ein Sheet zuerst

    # Parse alle Interviews
    parser.parse_all_interviews()

    # Interviewabschnitt drucken
    print(parser.data_structure["01"]["language"])
    df = parser.get_transcripts(episode_type='depression', include_time_keys=False)["01"]["depression"]
    print(df.iloc[0]['transcript'])

    # 2. Komplettes Transkript anschauen
    parser.print_patient_transcript('01')

    # 3. Nur Depression anschauen
    parser.print_patient_transcript('07', episode_type='mania')

    # 4. Nur aktuelle Episode anschauen
    parser.print_patient_transcript('01', episode_type='depression', time_frame='aktuelle')

    # 5. In JSON exportieren (für externe Bearbeitung)
    parser.export_patient_to_json('01', 'patient_01.json')

    # 6. Excel-Sheet inspizieren (rohe Daten)
    parser.inspect_sheet('Interview_01')

    # Speichere die Struktur
    parser.save_to_pickle('interview_data.pkl')

    # Zugriff auf die Daten
    for patient_id, patient_data in data_structure.items():
        print(f"\n=== Patient {patient_id} ===")
        print(f"Verdachtsdiagnose: {patient_data['insgesammt'].get('Verdachtsdiagnose')}")

        # Depression
        if 'aktuelle' in patient_data['interviews'].get('depression', {}):
            df_dep_a = patient_data['interviews']['depression']['aktuelle']
            print(f"Aktuelle Depression: {len(df_dep_a)} Chunks")
            criteria = patient_data['interviews']['depression']['Diagnostische Kriterien'].get('aktuelle', {})
            print(f"  Erfüllte Kriterien: {criteria.get('erfüllte Kriterien', [])}")

        # Mania
        if 'aktuelle' in patient_data['interviews'].get('mania', {}):
            df_mania_a = patient_data['interviews']['mania']['aktuelle']
            print(f"Aktuelle Mania: {df_mania_a} ")
#"""
