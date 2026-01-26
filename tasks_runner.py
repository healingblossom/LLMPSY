"""
Task Runner für LLM-PSY Studie.
Verwaltet Task-Ausführung, Dependencies, Datenverwaltung und Ergebnisspeicherung.
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd

from prompts.prompt_builder import (
    AlpacaPromptBuilder,
    MistralPromptBuilder,
    FlanPromptBuilder,
    OpenrouterPromptBuilder
)
from model_manager import ModelManager
from utils.data_loader import InterviewDataParser

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class DependencyMode(Enum):
    """Modus für Task-Dependencies: Ground-Truth vs. LLM-Results"""
    GROUND_TRUTH = "ground_truth"  # Nutze Ground-Truth aus Daten
    DEPENDENCY = "dependency"  # Nutze Ergebnisse vorheriger Tasks


@dataclass
class TaskResult:
    """Speichert Ergebnis einer Task-Ausführung für einen Patient+Episode"""
    task_id: str
    patient_id: str
    episode_type: Optional[str]  # None für nicht episode-divided Tasks
    variant_name: str
    run_number: int  # Welche Replikation (1, 2, ..., x)
    prompt: str
    result: str
    timestamp: str
    model_name: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# TASK RUNNER KLASSE
# ============================================================================

class TaskRunner:
    """
    Zentrale Klasse für Task-Ausführung mit folgenden Features:
    - Selektive Task-Ausführung für bestimmte Patienten
    - Dependency-Validierung (Ground-Truth vs. LLM-Results)
    - Strukturierte Ergebnisspeicherung
    - Mehrfache Generierungen pro Task (für Reproduzierbarkeit)
    - Dynamische Dateneinpflanzung in Prompts
    """

    def __init__( self, config_path: str = "config/config.yaml", data_file: str = None, verbose: bool = False):
        """
        Initialisiere TaskRunner

        Args:
            config_path: Pfad zur config.yaml
            data_file: Pfad zur Interview-Datei (Excel)
            verbose: Debug-Output
        """
        self.verbose = verbose
        self.config_path = config_path

        # Lade Konfigurationen
        self._load_config(config_path)

        # Initialisiere Data Parser
        if data_file:
            self.data_parser = InterviewDataParser(data_file, verbose=verbose)
            self.data_parser.parse_all_interviews()
        else:
            self.data_parser = None
            print("Fehler: Es fehlt eine Datei. Ergänze den parameter data_file mit einem Pfad zur Interview-Datei zum Funktionsaufruf!")

        # Task-Abhängigkeiten (aus tasks.yaml)
        self._build_dependency_graph()

        # Ergebnisse Cache (für aktuelle Session)
        self.results_cache: Dict[str, List[TaskResult]] = {}

        # Lade alle Prompts beim Starten (optional, kann auch lazy geladen werden)
        self.prompt_builders: Dict[str, Any] = {}
        self._initialize_prompt_builders()

        logger.info("TaskRunner initialisiert")

    # ========================================================================
    #             INITIALIZATION METHODS
    # ========================================================================

    def _load_config(self, config_path: str) -> None:
        """Lade config.yaml und tasks.yaml"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Bestimme tasks.yaml Pfad (usually neben config.yaml)
        config_dir = os.path.dirname(config_path)
        tasks_yaml_path = os.path.join(config_dir, "tasks.yaml")

        if not os.path.exists(tasks_yaml_path):
            # Fallback: suche im Projekt-Verzeichnis
            proj_dir = self.config.get('paths', {}).get('proj_directory', '.')
            tasks_yaml_path = os.path.join(proj_dir, "config", "tasks.yaml")

        with open(tasks_yaml_path, 'r', encoding='utf-8') as f:
            self.tasks_config = yaml.safe_load(f)

        logger.info(f"Konfigurationen geladen von {config_path}")

    def _build_dependency_graph(self) -> None:
        """
        Baue Abhängigkeitsgraph aus tasks.yaml

        Format:
        {
            'task_1': {'depends_on': []},
            'task_2': {'depends_on': ['task_1']},
            'task_3': {'depends_on': ['task_2']},
            'task_4': {'depends_on': ['task_1', 'task_2', 'task_3']},
        }
        """
        self.dependency_graph: Dict[str, Dict[str, Any]] = {}

        for task in self.tasks_config.get('tasks', []):
            task_id = task.get('task_id')
            snippet_config = task.get('snippet_config', {})

            # Analysiere inputs, um Dependencies zu finden
            inputs = snippet_config.get('inputs', [])
            depends_on = []

            if 'input_from_task1' in inputs:
                depends_on.append('task_1_symptom_detection_and_sectioning')
            if 'input_from_task2' in inputs:
                depends_on.append('task_2_diagnostic_criteria')
            if 'input_from_task3' in inputs:
                depends_on.append('task_3_diagnostic')
            if 'input_from_task4' in inputs:
                depends_on.append('task_4_summary')

            self.dependency_graph[task_id] = {
                'task_config': task,
                'depends_on': depends_on,
                'episodes_divided': snippet_config.get('episodes_divided', False),
            }

        if self.verbose:
            logger.info(f"Dependency-Graph gebaut: {self.dependency_graph}")

    def _initialize_prompt_builders(self) -> None:
        """Initialisiere PromptBuilder für alle bekannten Formate"""
        builder_classes = {
            'alpaca': AlpacaPromptBuilder,
            'mistral': MistralPromptBuilder,
            'flan': FlanPromptBuilder,
            'openrouter': OpenrouterPromptBuilder,
        }

        for format_name, builder_class in builder_classes.items():
            try:
                self.prompt_builders[format_name] = builder_class()
                logger.info(f"PromptBuilder '{format_name}' initialisiert")
            except Exception as e:
                logger.warning(f"Konnte PromptBuilder '{format_name}' nicht initialisieren: {e}")


    # ========================================================================
    #           DEPENDENCY & VALIDATION METHODS
    # ========================================================================

    def _validate_dependencies(self, requested_task_ids: List[str]) -> None:
        """
        Validiere, dass alle Dependencies für die Tasks erfüllt sind.

        Raises:
            ValueError: Wenn eine Task angefordert wird, deren Dependencies nicht erfüllt sind
        """
        logger.info("Validiere Task-Dependencies...")

        for task_id in requested_task_ids:
            task_config = self.dependency_graph.get(task_id)

            if not task_config:
                raise ValueError(f"Task '{task_id}' nicht in tasks.yaml definiert")

            depends_on = task_config.get('depends_on', [])

            for dep_task_id in depends_on:
                if dep_task_id not in requested_task_ids:
                    raise ValueError(
                        f"Task '{task_id}' hängt von '{dep_task_id}' ab, "
                        f"aber '{dep_task_id}' wurde nicht in den angeforderten Tasks aufgeführt!\n"
                        f"Bitte addiere '{dep_task_id}' zu deiner Task-Liste."
                    )

        logger.info("✓ Alle Dependencies validiert")

    # ========================================================================
    # DATA & PROMPT FORMATTING METHODS
    # ========================================================================

    def _get_available_episodetypes_for_patient(self, patient_id: str, episodes_divided: bool) -> List[Optional[str]]:
        """
        Bestimme welche Episodes für einen Patienten relevant sind. Gibt nur zurück, welche Episodentypen ein Patient hat, nicht die Interviews selbst.

        Returns:
            ['depression', 'mania'] wenn episodes_divided=True
            [None] wenn episodes_divided=False (gesamter Patient)
        """
        if not episodes_divided:
            return [None]

        # Hole alle Depression/Mania Episodes für Patienten
        episodes = []

        patient_data = self.data_parser.data_structure.get(patient_id, {})
        interviews = patient_data.get('interviews', {})

        if 'depression' in interviews and interviews['depression']:
            episodes.append('depression')
        if 'mania' in interviews and interviews['mania']:
            episodes.append('mania')

        return episodes if episodes else [None]

    def _get_prompt_variants( self, prompts_dict: Dict, task_id: str, episode_type: Optional[str] ) -> Dict[str, str]:
        """
        Holt selektiv Prompt-Variants für eine Task+Episode Kombination aus einem Prompt_dict.

        Returns:
            {variant_name -> prompt_string}
        """
        task_prompts = prompts_dict.get(task_id, {})

        if not task_prompts:
            logger.warning(f"Keine Prompts gefunden für Task: {task_id}")
            return {}

        # Filter nach Episode wenn relevant
        filtered_prompts = {}
        for variant_name, prompt_text in task_prompts.items():
            # Variant names haben Format: "task_X_all_episodes_role_TYPE_examples_TYPE"
            #                        oder: "task_X_EPISODE_TYPE_role_TYPE_examples_TYPE"

            if episode_type is None:
                # Für non-episode-divided Tasks: nutze die mit "all_episodes"
                if "all_episodes" in variant_name:
                    filtered_prompts[variant_name] = prompt_text
            else:
                # Für episode-divided Tasks: filter nach Episode
                if episode_type in variant_name:
                    filtered_prompts[variant_name] = prompt_text

        return filtered_prompts if filtered_prompts else task_prompts

    def _format_prompt_with_data( self, prompt_template: str, task_id: str,  patient_id: str, episode_type: Optional[str], dependency_mode: DependencyMode) -> str:
        """
        Fülle Prompt-Template mit echten Daten ein.

        Platzhalter:
        - {transcript}: Roher Interview-Text
        - {previous_results_task1}: Ergebnisse von Task 1
        - {previous_results_task2}: Ergebnisse von Task 2
        - {previous_results_task3}: Ergebnisse von Task 3
        """
        formatted_prompt = prompt_template

        # ====== {transcript} ======
        if "{transcript}" in formatted_prompt:
            transcript = self._get_transcript_data(patient_id, episode_type)
            formatted_prompt = formatted_prompt.replace("{transcript}", transcript)

        # ====== {previous_results_task1} ======
        if "{previous_results_task1}" in formatted_prompt:
            if dependency_mode == DependencyMode.GROUND_TRUTH:
                task1_results = self._get_ground_truth_task1(patient_id, episode_type)
            else:
                task1_results = self._get_llm_results_task1(patient_id, episode_type)

            formatted_prompt = formatted_prompt.replace(
                "{previous_results_task1}",
                task1_results
            )

        # ====== {previous_results_task2} ======
        if "{previous_results_task2}" in formatted_prompt:
            if dependency_mode == DependencyMode.GROUND_TRUTH:
                task2_results = self._get_ground_truth_task2(patient_id, episode_type)
            else:
                task2_results = self._get_llm_results_task2(patient_id, episode_type)

            formatted_prompt = formatted_prompt.replace(
                "{previous_results_task2}",
                task2_results
            )

        # ====== {previous_results_task3} ======
        if "{previous_results_task3}" in formatted_prompt:
            if dependency_mode == DependencyMode.GROUND_TRUTH:
                task3_results = self._get_ground_truth_task3(patient_id)
            else:
                task3_results = self._get_llm_results_task3(patient_id)

            formatted_prompt = formatted_prompt.replace(
                "{previous_results_task3}",
                task3_results
            )

        return formatted_prompt

    def _get_transcript_data(self, patient_id: str, episode_type: Optional[str]) -> str:
        """Hole Rohen Interview-Text für Patient+Episode"""
        if not self.data_parser:
            return "[TRANSKRIPT NICHT VERFÜGBAR]"

        try:
            if episode_type is None:
                # Für non-episode-divided: kombiniere alle Episodes
                all_transcripts = []
                transcripts = self.data_parser.get_transcripts()
                patient_data = transcripts.get(patient_id, {})

                for disorder in ['depression', 'mania']:
                    for time_frame in ['aktuelle', 'frühere']:
                        df = patient_data.get(disorder, {}).get(time_frame)
                        if isinstance(df, pd.DataFrame) and len(df) > 0:
                            all_transcripts.append(df['transcript'].str.cat(sep='\n\n'))

                return '\n\n---\n\n'.join(all_transcripts)
            else:
                # Für episode-divided: hole spezifische Episode
                transcripts = self.data_parser.get_transcripts(episode_type)
                patient_transcripts = transcripts.get(patient_id, {})

                combined = []
                for time_frame in ['aktuelle', 'frühere']:
                    df = patient_transcripts.get(time_frame)
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        combined.append(df['transcript'].str.cat(sep='\n\n'))

                return '\n\n---\n\n'.join(combined)

        except Exception as e:
            logger.error(f"Fehler beim Laden Transkript für Patient {patient_id}: {e}")
            return "[TRANSKRIPT FEHLGESCHLAGEN]"

    def _get_ground_truth_task1(self, patient_id: str, episode_type: Optional[str]) -> str:
        """Hole Ground-Truth Symptome aus Daten für Task 1"""
        if not self.data_parser:
            return "[]"

        try:
            if episode_type:
                symptoms = self.data_parser.get_symptoms(episode_type)
                patient_symptoms = symptoms.get(patient_id, {})
            else:
                symptoms = self.data_parser.get_symptoms()
                patient_symptoms = symptoms.get(patient_id, {})

            # Formatiere als JSON
            result_json = json.dumps(patient_symptoms, ensure_ascii=False, indent=2)
            return result_json

        except Exception as e:
            logger.error(f"Fehler beim Laden Ground-Truth Task1: {e}")
            return "[]"

    def _get_llm_results_task1(self, patient_id: str, episode_type: Optional[str]) -> str:
        """Hole LLM-Ergebnisse von Task 1 aus Cache"""
        cache_key = f"task_1_symptom_detection_and_sectioning_{patient_id}_{episode_type}"

        if cache_key not in self.results_cache:
            logger.warning(f"Task1 Ergebnisse nicht im Cache: {cache_key}")
            return "[]"

        # Nutze neueste Ergebnisse (höchste run_number)
        results = self.results_cache[cache_key]
        latest_result = max(results, key=lambda r: r.run_number)

        return latest_result.result

    def _get_ground_truth_task2(self, patient_id: str) -> str:
        """Hole Ground-Truth Diagnosekriterien aus Daten für Task 2"""
        if not self.data_parser:
            return "{}"

        try:
            criteria = self.data_parser.get_diagnostic_criteria()
            patient_criteria = criteria.get(patient_id, {})

            result_json = json.dumps(patient_criteria, ensure_ascii=False, indent=2)
            return result_json

        except Exception as e:
            logger.error(f"Fehler beim Laden Ground-Truth Task2: {e}")
            return "{}"

    def _get_llm_results_task2(self, patient_id: str, episode_type: Optional[str]) -> str:
        """Hole LLM-Ergebnisse von Task 2 aus Cache"""
        cache_key = f"task_2_diagnostic_criteria_{patient_id}_{episode_type}"

        if cache_key not in self.results_cache:
            logger.warning(f"Task2 Ergebnisse nicht im Cache: {cache_key}")
            return "{}"

        results = self.results_cache[cache_key]
        latest_result = max(results, key=lambda r: r.run_number)

        return latest_result.result

    def _get_ground_truth_task3(self, patient_id: str) -> str:
        """Hole Ground-Truth Diagnose aus Daten für Task 3"""
        if not self.data_parser:
            return "{}"

        try:
            summary = self.data_parser.get_summary_data(patient_ids=[patient_id], groups=['Verdachtsdiagnose'])
            diagnose = summary.get('Verdachtsdiagnose', {}).get(patient_id)

            return json.dumps({"Diagnose": diagnose}, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Fehler beim Laden Ground-Truth Task3: {e}")
            return "{}"

    def _get_llm_results_task3(self, patient_id: str) -> str:
        """Hole LLM-Ergebnisse von Task 3 aus Cache"""
        cache_key = f"task_3_diagnostic_{patient_id}_None"

        if cache_key not in self.results_cache:
            logger.warning(f"Task3 Ergebnisse nicht im Cache: {cache_key}")
            return "{}"

        results = self.results_cache[cache_key]
        latest_result = max(results, key=lambda r: r.run_number)

        return latest_result.result

    # ========================================================================
    #             MODEL & PROMPT LOADING METHODS
    # ========================================================================

    def _load_model(self, model_name: str) -> Any:
        """Lade Modell via ModelManager"""
        manager = ModelManager(self.config_path)
        return manager.load_model(model_name)

    def _get_prompts_for_model(self, model_name: str) -> Dict:
        """
        Lade alle Prompts für ein Modell

        Returns:
            {task_id -> {variant_name -> prompt_string}}
        """
        # Finde Model-Spec
        model_spec = None
        for spec in self.config.get('models', []):
            if spec.get('name') == model_name:
                model_spec = spec
                break

        if not model_spec:
            raise ValueError(f"Modell '{model_name}' nicht in config.yaml gefunden")

        format_name = model_spec.get('format')

        # Hole PromptBuilder für diesen Format
        if format_name not in self.prompt_builders:
            raise ValueError(f"PromptBuilder für Format '{format_name}' nicht initialisiert")

        builder = self.prompt_builders[format_name]

        # Baue alle Prompts
        all_prompts = builder.build_all_prompts()

        return all_prompts

    # ========================================================================
    #                  RESULTS STORAGE METHODS
    # ========================================================================

    def _save_results( self, all_results: Dict[str, List[TaskResult]], output_dir: str, model_name: str ) -> None:
        """
        Speichere Ergebnisse strukturiert:

        results/
        ├── {model_name}/
        │   ├── {task_id}/
        │   │   ├── results.jsonl (alle TaskResult objects)
        │   │   ├── patient_01/
        │   │   │   ├── depression_variant_1_run_1.json
        │   │   │   ├── depression_variant_1_run_2.json
        │   │   │   └── mania_variant_1_run_1.json
        │   │   └── patient_02/
        │   │       └── ...
        │   └── task_id_2/
        │       └── ...
        │   └── summary.json (Metadaten)
        """
        # Hauptverzeichnis für Modell
        model_results_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

        for task_id, task_results in all_results.items():
            task_dir = os.path.join(model_results_dir, task_id)
            os.makedirs(task_dir, exist_ok=True)

            # Speichere alle Ergebnisse als JSONL (für einfache Weiterverarbeitung)
            results_jsonl_path = os.path.join(task_dir, f"results_{timestamp}.jsonl")
            with open(results_jsonl_path, 'w', encoding='utf-8') as f:
                for result in task_results:
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')

            logger.info(f"  Gespeichert JSONL: {results_jsonl_path}")

            # Speichere auch pro Patient/Episode (optional, für schnelle Zugriffe)
            for result in task_results:
                patient_dir = os.path.join(task_dir, result.patient_id)
                os.makedirs(patient_dir, exist_ok=True)

                # Dateiname: {episode}_{variant}_{run}.json
                episode_str = result.episode_type or "all_episodes"
                filename = f"{episode_str}_{result.variant_name}_run_{result.run_number}.json"
                filepath = os.path.join(patient_dir, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        # Speichere Summary/Metadaten
        summary = {
            'timestamp': timestamp,
            'model_name': model_name,
            'total_tasks': len(all_results),
            'total_results': sum(len(r) for r in all_results.values()),
            'task_counts': {task_id: len(results) for task_id, results in all_results.items()},
        }

        summary_path = os.path.join(model_results_dir, f"summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Summary gespeichert: {summary_path}")


    # ========================================================================
    #            METHODE ZUR VERWENDUNG AUẞERHALB DER KLASSE (SCHNITTSTELLE)
    # ========================================================================

    def run_tasks( self, model_name: str, task_ids: Optional[List[str]] = None, patient_ids: Optional[List[str]] = None, dependency_mode: DependencyMode = DependencyMode.GROUND_TRUTH, num_runs: int = 3, output_dir: Optional[str] = None) -> Dict[str, List[TaskResult]]:
        """
        Führe Tasks.

        Args:
            task_ids: Liste von Task-IDs zum Ausführen (z.B. ['task_1_...', 'task_2_...'])
                     Wenn None: Alle Tasks in Reihenfolge
            patient_ids: Liste von Patient-IDs (z.B. ['01', '02', '03'])
                        Wenn None: Alle Patienten
            model_name: Modellname aus config.yaml (z.B. 'mental_alpaca')
            dependency_mode: GROUND_TRUTH oder DEPENDENCY
            num_runs: Wie oft jede Task repliziert werden soll (für Reproduzierbarkeit)
            output_dir: Verzeichnis für Ergebnisspeicherung
                       Wenn None: Verwendet config['paths']['proj_directory']/results

        Returns:
            Dict[task_id -> List[TaskResult]]

        Raises:
            ValueError: Bei Dependency-Validierungsfehlern
        """
        # Validierung
        if task_ids is None:
            task_ids = [t['task_id'] for t in self.tasks_config.get('tasks', [])]

        self._validate_dependencies(task_ids)

        if patient_ids is None:
            if self.data_parser:
                patient_ids = list(self.data_parser.data_structure.keys())
            else:
                raise ValueError("Keine patient_ids angegeben und kein Data Parser initialisiert")

        # Setup
        if output_dir is None:
            proj_dir = self.config.get('paths', {}).get('proj_directory', '.')
            output_dir = os.path.join(proj_dir, 'results')

        os.makedirs(output_dir, exist_ok=True)

        # Lade Modell
        logger.info(f"Lade Modell: {model_name}")
        model = self._load_model(model_name)

        # Lade Prompts
        logger.info(f"Lade Prompts für Modell: {model_name}")
        prompts_dict = self._get_prompts_for_model(model_name)

        # Hauptschleife: Tasks → Patienten → Episodes → Runs
        all_results: Dict[str, List[TaskResult]] = {}

        for task_id in task_ids:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Starte Task: {task_id}")
            logger.info(f"{'=' * 60}")

            task_results = []

            # Hole Task-Konfiguration
            task_config = self.dependency_graph.get(task_id)
            if not task_config:
                logger.error(f"Task {task_id} nicht gefunden in tasks.yaml")
                continue

            episodes_divided = task_config['episodes_divided']

            # Iteriere über Patienten
            for patient_id in patient_ids:
                logger.info(f"\n  Patient: {patient_id}")

                # Bestimme Episodes für diesen Patienten
                episodes = self._get_available_episodetypes_for_patient(
                    patient_id,
                    episodes_divided
                )

                # Für jede Episode (oder nur eine wenn nicht episodes_divided)
                for episode_type in episodes:
                    logger.info(f"    Episode: {episode_type or 'N/A'}")

                    # Hole Prompts für diese Kombination
                    prompt_variants = self._get_prompt_variants(
                        prompts_dict,
                        task_id,
                        episode_type
                    )

                    # Für jeden Prompt-Variant
                    for variant_name, prompt_template in prompt_variants.items():
                        logger.info(f"      Variant: {variant_name}")

                        # Fülle Prompt mit Daten
                        formatted_prompt = self._format_prompt_with_data(
                            prompt_template,
                            task_id,
                            patient_id,
                            episode_type,
                            dependency_mode
                        )

                        # Führe x Mal aus (für Reproduzierbarkeit)
                        for run_num in range(1, num_runs + 1):
                            logger.info(f"        Run {run_num}/{num_runs}")

                            # Generiere mit Modell
                            result_text = model.generate_from_prompt(formatted_prompt)

                            # Speichere Result-Objekt
                            task_result = TaskResult(
                                task_id=task_id,
                                patient_id=patient_id,
                                episode_type=episode_type,
                                variant_name=variant_name,
                                run_number=run_num,
                                prompt=formatted_prompt,
                                result=result_text,
                                timestamp=datetime.now().isoformat(),
                                model_name=model_name
                            )

                            task_results.append(task_result)

                            # Cache für Dependencies
                            cache_key = f"{task_id}_{patient_id}_{episode_type}"
                            if cache_key not in self.results_cache:
                                self.results_cache[cache_key] = []
                            self.results_cache[cache_key].append(task_result)

            all_results[task_id] = task_results

        # Speichere alle Ergebnisse
        self._save_results(all_results, output_dir, model_name)

        logger.info(f"\n✓ Alle Tasks abgeschlossen!")
        logger.info(f"Ergebnisse gespeichert in: {output_dir}")

        return all_results

# ============================================================================
#             CONVENIENCE FUNCTIONS (Beispiel_nutzen)
# ============================================================================

def run_default_pipeline( model_name: str = "mental_alpaca", patient_ids: Optional[List[str]] = None, num_runs: int = 1, use_ground_truth: bool = True) -> Dict[str, List[TaskResult]]:
    """
    Convenience-Funktion: Führe alle Tasks für alle Patienten aus.

    Args:
        model_name: Modell zu verwenden
        patient_ids: Patient-IDs (None = alle)
        num_runs: Anzahl Replikationen pro Task
        use_ground_truth: True = Ground-Truth Dependencies, False = LLM Dependencies

    Returns:
        Alle Ergebnisse

    Beispiel:
        results = run_default_pipeline(
            model_name="mental_alpaca",
            patient_ids=['01', '02'],
            num_runs=3,
            use_ground_truth=True
        )
    """
    runner = TaskRunner()

    dependency_mode = (
        DependencyMode.GROUND_TRUTH if use_ground_truth
        else DependencyMode.DEPENDENCY
    )

    return runner.run_tasks(
        task_ids=None,  # Alle Tasks
        patient_ids=patient_ids,
        model_name=model_name,
        dependency_mode=dependency_mode,
        num_runs=num_runs,
    )


if __name__ == "__main__":
    # ========================================================================
    # BEISPIELE ZUR NUTZUNG
    # ========================================================================

    # Beispiel 1: Einfache Ausführung aller Tasks
    # results = run_default_pipeline(
    #     model_name="mental_alpaca",
    #     patient_ids=['01', '02', '03'],
    #     num_runs=3
    # )

    # Beispiel 2: Nur Task 1 und Task 2 für einen Patienten
    # runner = TaskRunner()
    # results = runner.run_tasks(
    #     task_ids=['task_1_symptom_detection_and_sectioning', 'task_2_diagnostic_criteria'],
    #     patient_ids=['01'],
    #     model_name='mental_alpaca',
    #     num_runs=2,
    # )

    # Beispiel 3: Mit LLM-Dependencies statt Ground-Truth
    # runner = TaskRunner()
    # results = runner.run_tasks(
    #     task_ids=None,  # Alle
    #     patient_ids=None,  # Alle
    #     model_name='mistral',
    #     dependency_mode=DependencyMode.DEPENDENCY,
    #     num_runs=1,
    # )

    print("TaskRunner bitte über TaskRunner.run_tasks() aufrufen!")
