# task_prompts.py alle Prompt-Bausteine zum Zusammensetzen der tasks

"""
Modulare Prompt-Bausteine für Tasks.
Wie sich die Prompts zusammensetzen, ist in tasks.yaml definiert
"""

# ============================================================================
# KOMPONENTE: Role (Variierend)
# ============================================================================

ROLE_PSYCHIATRIST = "Du bist ein erfahrener Psychiater."
ROLE_NONE = ""

ROLES = {
    "role": ROLE_PSYCHIATRIST,
    "none": ROLE_NONE,
}

# ============================================================================
# KOMPONENTE: Chain-of-Thought (Optional)
# ============================================================================

COT_ACTIVE = "Erledige die Aufgabe Schritt für Schritt und dokumentiere alles."

COT = {
    "cot": COT_ACTIVE
}


# ============================================================================
# KOMPONENTE: Aufgabenbeschreibungen
# ============================================================================

TASK1_EXTRACT_SYMPTOMS = (
    "Identifiziere psychiatrisch relevante Symptome für Major Depression Episode und Bipolare Störungen (DSM-5), und zitiere die jeweiligen Abschnitte (Wortlaute), die jedes Symptom belegen."
)
TASK4_SUMMARY = (
    "Fasse den Fall des Patienten  zusammen. Verwende hierfür die  Symptome, Abschnitte und erfüllten Diagnosekriterien aus dem Input."
    "Deine Aufgabe ist es:\
    1. wichtige biografische Elemente zusammenzufassen\
    2. die Symptome und erfüllte Diagnosekriterien darzustellen"
)

TASK2_DIAGNOSTIC_CRITERIA_MAPPING = ("")
TASK3_DIAGNOSIS_PREDICTION = ("")


TASKS = {
    "task1_extract_symptoms": TASK1_EXTRACT_SYMPTOMS,
    "task4_summary": TASK4_SUMMARY,
    "task2_diagnostic_criteria_mapping": TASK2_DIAGNOSTIC_CRITERIA_MAPPING,#Todo
    "task3_diagnosis_prediction": TASK3_DIAGNOSIS_PREDICTION,#todo
}

# ============================================================================
# KOMPONENTE: Output-Format
# ============================================================================

FORMAT_SYMPTOMS_JSON = (
    "Gib deine Antwort ausschließlich in folgendem Format als gültiges JSON zurück: [{\"symptom\": \"<label>\", \"section\": \"<exact transcript quote>\"}, {\"symptom\": \"<label>\", \"section\": \"<exact transcript quote>\"}, ...].\\"
    "Verwende ausschließlich folgende Symptome:\\"
    "{symptom}\\"
    "Zitiere die Abschnitte aus dem Transkript korrekt. Verändere dabei nichts. Gib nur die minimal nötigen Wörter zur Belegung des Symptoms (keine ganzen Sätze). Achte darauf, das der Inhalt verständlich ist.\\"
    "Wenn ein Abschnitt mehrere Symptome enthält, liste sie zusammen auf: [{\"symptom\": \"symptom1, symptom2\", \"section\": \"...\"}].\\"
    "Beziehe dich nur auf Symptome, die die Patient:in hat. Ignoriere Symptome, die die Patient:in berichtet, NICHT zu haben.\\"
    "Wenn keine psychiatrisch relevanten Symptome vorliegen, antworte mit: [{\"symptom\": \"none\", \"section\": \"none\"}]."
)

SYMPTOME_DEPRESSIVE_EPISODE = "Depressive Symptome: Depressive Stimmung, Interessen- oder Freudeverlust, Verminderter oder gesteigerter Appetit, Gewichtsverlust oder -zunahme, Insomnie oder Hypersomnie, Psychomotorische Unruhe oder Verlangsamung, Müdigkeit oder Energieverlust, Schuld- oder Wertlosigkeitsgefühle, Konzentrations- oder Entscheidungsschwierigkeiten, Suizidalität, Psychotische Symptome"
SYMPTOME_MANIC_EPISODE = "Hypomanische oder Manische Symptome: Gereizte oder gehobene oder expansive Stimmung, Gesteigertes Selbstwertgefühl oder Größenideen, Vermindertes Schlafbedürfnis, Gesteigertes Sprechbedürfnis oder Rededruck, Ideenflucht oder Gedankenrasen, Ablenkbarkeit, Gesteigerte zielgerichtete Aktivität oder psychomotorische Unruhe, Risikoverhalten, Psychotische Symptome"

FORMAT_SUMMARY = (
    "Die Zusammenfassung soll maximal 250 Wörtern haben.\
    Folge diesem Format und füge die Patientenbedingten Informationen in die <Text>-Bereiche:\
    Biografische Zusammenfassung: <Text>\
    Symptome und Diagnosekriterien: <Text>\
    Formuliere es prägnant, klinisch und begründe deine Überlegungen. Wenn die Hinweise unzureichend sind oder du dir unsicher bist, gib dies an.\
    Wenn du biografische Elemente weitergibst, nutze den Konjunktiv oder Zitiere. Alle Inhalte müssen aus den Input."
)

FORMAT_DIAGNOSTIC_CRITERIA= ("{criteria}")
CRITERIA_DEPRESSIVE_EPISODE = "" #TODO
CRITERIA_MANIC_EPISODE = "" #TODO

FORMAT_DIAGNOSIS= ("")

FORMATS = {
    "task1_symptom_format": FORMAT_SYMPTOMS_JSON,
    "task4_summary_format": FORMAT_SUMMARY,
    "task2_diagnostic_criteria_format": FORMAT_DIAGNOSTIC_CRITERIA,#TODO
    "task3_diagnosis_format": FORMAT_DIAGNOSIS,#TODO
}

SYMPTOMS = {
    "depression": SYMPTOME_DEPRESSIVE_EPISODE,
    "mania": SYMPTOME_MANIC_EPISODE,
}
CRITERIA = {
    "depression": CRITERIA_DEPRESSIVE_EPISODE,
    "mania": CRITERIA_MANIC_EPISODE,
}
# ============================================================================
# KOMPONENTE: Prompt-Varianten (Zero/One/Few-Shot)
# ============================================================================

EXAMPLES_NONE = ""

EXAMPLES_ONESHOT = (
    "Hier ist ein Beispiel:\n\n"
    #todo beispiel fehlt
)

EXAMPLES_FEWSHOT = (
    "Hier sind mehrere Beispiele:\n\n"
    #todo beispiele fehlen
)

EXAMPLES = {
    "zeroshot": EXAMPLES_NONE,
    "oneshot": EXAMPLES_ONESHOT,
    "fewshot": EXAMPLES_FEWSHOT,
}

# ============================================================================
# KOMPONENTE: Input-Text (Platzhalter)
# ============================================================================

INPUT_TRANSCRIPT = (
    "Hier ist das Interview-Transkript:\n\n"
    "{transcript}"
)

INPUT_FROM_TASK1 = (
    "Das sind die Symptome:\n\n"
    "{previous_results_task1}"
)

INPUT_FROM_TASK2 = (
    "Das sind die Diagnosekriterien:\n\n"
    "{previous_results_task2}"
)

INPUT_FROM_TASK3 = (
    "Das ist die Diagnose: {previous_results_task3}"
)

INPUTS = {
    "transcript": INPUT_TRANSCRIPT,
    "input_from_task1": INPUT_FROM_TASK1,
    "input_from_task2": INPUT_FROM_TASK2,
    "input_from_task3": INPUT_FROM_TASK3,
}

