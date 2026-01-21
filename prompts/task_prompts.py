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

COT_ACTIVE = "Begründe deine Entscheidungen zum Erledigen folgender Aufgabe sorgfältig."

COT = {
    "cot": COT_ACTIVE
}


# ============================================================================
# KOMPONENTE: Aufgabenbeschreibungen
# ============================================================================

TASK1_EXTRACT_SYMPTOMS = (
    "Identifiziere psychiatrisch relevante Symptome für Major Depression Episode oder Bipolare Störungen (DSM-5), und zitiere die jeweiligen Abschnitte (Wortlaute), die jedes Symptom belegen.\n"
)
TASK4_SUMMARY = (
    "Fasse den Fall des Patienten  zusammen. Verwende hierfür die  Symptome, Abschnitte und erfüllten Diagnosekriterien aus dem Input. \n"
    "Deine Aufgabe ist es:\n"
    "1. wichtige biografische Elemente zusammenzufassen\n"
    "2. die Symptome und erfüllte Diagnosekriterien darzustellen.\n"
)

TASK2_DIAGNOSTIC_CRITERIA_MAPPING = ("Prüfe, ob die DSM-5-Diagnosekriterien für eine Major Depression (MDD) oder eine Bipolare Störung (BD) erfüllt sind.\n")
TASK3_DIAGNOSIS_PREDICTION = ("Erstelle eine Verdachtsdiagnose für gegebenen Patient:in.\n")


TASKS = {
    "task1_extract_symptoms": TASK1_EXTRACT_SYMPTOMS,
    "task4_summary": TASK4_SUMMARY,
    "task2_diagnostic_criteria_mapping": TASK2_DIAGNOSTIC_CRITERIA_MAPPING,
    "task3_diagnosis_prediction": TASK3_DIAGNOSIS_PREDICTION,
}

# ============================================================================
# KOMPONENTE: Output-Format
# ============================================================================

FORMAT_SYMPTOMS_JSON = (
    "Gib deine Antwort ausschließlich in folgendem Format als gültiges JSON zurück: [{\"symptom\": \"<label>\", \"section\": \"<exact transcript quote>\"}, {\"symptom\": \"<label>\", \"section\": \"<exact transcript quote>\"}, ...].\n"
    "Verwende ausschließlich folgende Symptome:\n"
    "{symptom}\n"
    "Zitiere die Abschnitte aus dem Transkript korrekt. Verändere dabei nichts. Gib nur die minimal nötigen Wörter zur Belegung des Symptoms (keine ganzen Sätze). Achte darauf, das der Inhalt verständlich ist.\n"
    "Wenn ein Abschnitt mehrere Symptome enthält, liste sie zusammen auf: [{\"symptom\": \"symptom1, symptom2\", \"section\": \"...\"}].\n"
    "Beziehe dich nur auf Symptome, die die Patient:in hat. Ignoriere Symptome, die die Patient:in berichtet, NICHT zu haben.\n"
    "Wenn keine psychiatrisch relevanten Symptome vorliegen, antworte mit: [{\"symptom\": \"none\", \"section\": \"none\"}]."
)

SYMPTOME_DEPRESSIVE_EPISODE = "Depressive Symptome: Depressive Stimmung, Interessen- oder Freudeverlust, Verminderter oder gesteigerter Appetit, Gewichtsverlust oder -zunahme, Insomnie oder Hypersomnie, Psychomotorische Unruhe oder Verlangsamung, Müdigkeit oder Energieverlust, Schuld- oder Wertlosigkeitsgefühle, Konzentrations- oder Entscheidungsschwierigkeiten, Suizidalität, Psychotische Symptome"
SYMPTOME_MANIC_EPISODE = "Hypomanische oder Manische Symptome: Gereizte oder gehobene oder expansive Stimmung, Gesteigertes Selbstwertgefühl oder Größenideen, Vermindertes Schlafbedürfnis, Gesteigertes Sprechbedürfnis oder Rededruck, Ideenflucht oder Gedankenrasen, Ablenkbarkeit, Gesteigerte zielgerichtete Aktivität oder psychomotorische Unruhe, Risikoverhalten, Psychotische Symptome"

FORMAT_SUMMARY = (
    "Die Zusammenfassung soll maximal 250 Wörtern haben."
    "Folge diesem Format und füge die Patientenbedingten Informationen in die <Text>-Bereiche:\n"
    "Biografische Zusammenfassung: <Text>\n"
    "Symptome und Diagnosekriterien: <Text>\n"
    "Formuliere es prägnant, klinisch und begründe deine Überlegungen. Wenn die Hinweise unzureichend sind oder du dir unsicher bist, gib dies an.\
    Wenn du biografische Elemente weitergibst, nutze den Konjunktiv oder Zitiere. Alle Inhalte müssen aus dem Input ersichtlich sein."
)

FORMAT_DIAGNOSTIC_CRITERIA= ("Liste die erfüllten Kriterien exakt im folgenden Format auf:\n"
                             "{'ausgefüllte Diagnosekriterien': <Kriterium1>, <Kriterium2>, ...}\n"
                             "Nutze ausschließlich folgende Diagnosekriterien und fokussiere dich auf Informationen des Patienten:\n"
                             "{criteria}")

CRITERIA_DEPRESSIVE_EPISODE = "≥5 Symptome inkl. depressive Stimmung/Interessenverlust für ≥2 Wochen (siehe Symptome), Deutliches Leiden oder Beeinträchtigung, Keine Substanz-/medizinische Ursache, Keine Psychose, Keine manische/hypomanische Vorgeschichte"
CRITERIA_MANIC_EPISODE = "Manie: Stimmungs-/Aktivitätsveränderung ≥1 Woche oder Hospitalisierung (siehe Symptome), Manie: ≥3 Symptome in dieser Phase (siehe Symptome), Manie: Beeinträchtigung oder Hospitalisierung oder Psychose, Manie: Keine Substanz-/medizinische Ursache, Hypomanie: Stimmungs-/Aktivitätsveränderung ≥4 Tage (siehe Symptome), Hypomanie: ≥3 Symptome in dieser Phase (siehe Symptome), Hypomanie: Funktionsänderung, Hypomanie: Stimmungsveränderung für andere beobachtbar, Hypomanie: Keine Beeinträchtigung / keine Hospitalisierung / keine Psychose, Hypomanie: Keine Substanz-/medizinische Ursache"

FORMAT_DIAGNOSIS= ("Das sind vergebbare Diagnosen: "
                   "F30.0 Hypomanie, F30.1 Manie ohne psychotische Symptome, F30.2 Manie mit psychotischen Symptomen, F30.8 Sonstige manische Episoden, F30.9 Manische Episode, nicht näher bezeichnet, F31.0 Bipolare affektive Störung, gegenwärtig hypomanische Episode, F31.1 Bipolare affektive Störung, gegenwärtig manische Episode ohne psychotische Symptome, F31.2 Bipolare affektive Störung, gegenwärtig manische Episode mit psychotischen Symptomen, F31.3 Bipolare affektive Störung, gegenwärtig leichte oder mittelgradige depressive Episode, F31.4 Bipolare affektive Störung, gegenwärtig schwere depressive Episode ohne psychotische Symptome, F31.5 Bipolare affektive Störung, gegenwärtig schwere depressive Episode mit psychotischen Symptomen, F31.6 Bipolare affektive Störung, gegenwärtig gemischte Episode, F31.7 Bipolare affektive Störung, gegenwärtig remittiert, F31.8 Sonstige bipolare affektive Störungen, F31.9 Bipolare affektive Störung, nicht näher bezeichnet F32.0 Leichte depressive Episode, F32.1 Mittelgradige depressive Episode, F32.2 Schwere depressive Episode ohne psychotische Symptome, F32.3 Schwere depressive Episode mit psychotischen Symptomen, F32.8 Sonstige depressive Episoden, F32.9 Depressive Episode, nicht näher bezeichnet\n"
                   "Wähle die passendste Diagnose und Begründe diese sorgfältig")

FORMATS = {
    "task1_symptom_format": FORMAT_SYMPTOMS_JSON,
    "task4_summary_format": FORMAT_SUMMARY,
    "task2_diagnostic_criteria_format": FORMAT_DIAGNOSTIC_CRITERIA,
    "task3_diagnosis_format": FORMAT_DIAGNOSIS,
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
    "{beispiel}"
)

EXAMPLES_FEWSHOT = (
    "Hier sind mehrere Beispiele:\n\n"
    "{beispiel1}\n\n"
    "{beispiel2}\n\n"
    "{beispiel3}\n\n"
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
    "Das sind die Symptome von jeder Episode, die beschrieben wurde:\n\n"
    "{previous_results_task1}"
)

INPUT_FROM_TASK2 = (
    "Das sind Listen mit erfüllten Diagnosekriterien für jede Episode, die beschrieben wurde:\n\n"
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

