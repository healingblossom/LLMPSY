# pipeline_mental_alpaca.py 

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks_runner import run_all_tasks

if __name__ == "__main__":
    run_all_tasks(
        model_name="mental_alpaca",
        selected_tasks="task_1_symptom_detection_and_sectioning",  # Alle Tasks
        selected_variants=None  # Alle Varianten
    )