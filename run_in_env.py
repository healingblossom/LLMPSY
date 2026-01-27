# Skript um conda-Enviroment automatisiert für Modell zu wählen und auszuführen

import subprocess
import sys
import os
import yaml


def run_in_conda_env(env_name, python_script, conda_envs_root, script_args=None):
    """
    Führt ein Python-Script in einem conda-Environment aus, das
    in .conda_envs im Projekt liegt.

    Args:
        env_name: Name des Envs (z.B. "env_custom"), das unter .conda_envs/env_name liegt.
        python_script: Pfad zum Script relativ zum Projekt-Root.
        script_args: Optionale Argumente für das Script.

    Returns:
        int: Return-Code (0 = Erfolg, != 0 = Fehler)
        :param conda_envs_root: .conda_envs/ Pfad zu den conda-enviroments
    """
    
    if script_args is None:
        script_args = []

    env_prefix = os.path.join(conda_envs_root, env_name)

    cmd = [
        "conda", "run",
        "--prefix", env_prefix,
        "python", python_script
    ] + script_args

    print(f"\nStarte Script in Environment (Prefix): {env_prefix}")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nScript erfolgreich abgeschlossen")
    else:
        print(f"\nScript fehlgeschlagen (Exit-Code: {result.returncode})")

    return result.returncode


if __name__ == '__main__':
    print("="*60)
    print("Teste run_in_env_path")
    print("="*60)

    # Beispiel:
    exit_code = run_in_conda_env('gpulab_env', 'model_scripts/archiev_version_pipeline_mistral.py')

    if exit_code == 0:
        print("\nrun_in_conda_env_path funktioniert")
    else:
        print("\nFehler in run_in_conda_env_path")
        sys.exit(1)
