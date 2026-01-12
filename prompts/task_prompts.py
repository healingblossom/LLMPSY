# task_prompts.py alle Prompt-Bausteine zum Zusammensetzen der tasks

"""
Modulare Prompt-Bausteine für Tasks.
Wie sich die Prompts zusammensetzen, ist in tasks.yaml definiert
"""

# ============================================================================
# KOMPONENTE 1: Role (Optional, Variierend)
# ============================================================================

ROLE_PSYCHIATRIST = "Du bist ein erfahrener Psychiater mit Expertise in DSM-5 Diagnostik."
ROLE_NONE = ""

ROLES = {
    "role": ROLE_PSYCHIATRIST,
    "none": ROLE_NONE,
}

# ============================================================================
# KOMPONENTE 2: Aufgabenbeschreibung
# ============================================================================

TASK_EXTRACT_SYMPTOMS = (
    "Deine Aufgabe ist es, psychiatrisch relevante Symptome aus dem folgenden Interviewtranskript zu extrahieren. "
    "Konzentriere dich auf Symptome, die relevant für die Diagnose von Bipolaren Störungen und Schweren Depressiven Episoden sind."
)

TASKS = {
    "task1_extract_symptoms": TASK_EXTRACT_SYMPTOMS,
}

# ============================================================================
# KOMPONENTE 3: Output-Format
# ============================================================================

FORMAT_SYMPTOMS_JSON = (
    "Gib deine Antwort AUSSCHLIESSLICH als gültiges JSON-Array aus, "
    "mit folgendem Schema: "
    "[{'symptom': '<symptom_label>', 'section': '<exakte_zitierung_aus_text>'}, ...]"
)

FORMATS = {
    "task1_extract_symptoms": FORMAT_SYMPTOMS_JSON,
}

# ============================================================================
# KOMPONENTE 4: Symptom-Liste (Auswahl)
# ============================================================================

SYMPTOM_LIST_DEPRESSION_BIPOLAR = (
    "Wähle ausschliesslich aus folgenden Symptomen:\n\n"
    "Schwere Depressive Episode:\n"
    "- Depressive Stimmung\n"
    "- Interessen- oder Freudeverlust (Anhedonie)\n"
    "- Verminderter oder gesteigerter Appetit\n"
    "- Gewichtsverlust oder -zunahme\n"
    "- Insomnie oder Hypersomnie\n"
    "- Psychomotorische Unruhe oder Verlangsamung\n"
    "- Müdigkeit oder Energieverlust\n"
    "- Schuld- oder Wertlosigkeitsgefühle\n"
    "- Konzentrations- oder Entscheidungsschwierigkeiten\n"
    "- Suizidalität\n\n"
    "Manische oder Hypomanische Episode:\n"
    "- Gereizte, gehobene oder expansive Stimmung\n"
    "- Gesteigertes Selbstwertgefühl oder Größenideen\n"
    "- Vermindertes Schlafbedürfnis\n"
    "- Gesteigertes Sprechbedürfnis oder Rededruck\n"
    "- Ideenflucht oder Gedankenrasen\n"
    "- Ablenkbarkeit\n"
    "- Gesteigerte zielgerichtete Aktivität oder psychomotorische Unruhe\n"
    "- Risikoverhalten"
)

SYMPTOM_LISTS = {
    "task1_extract_symptoms": SYMPTOM_LIST_DEPRESSION_BIPOLAR,
}

# ============================================================================
# KOMPONENTE 5: Prompt-Varianten (Zero/One/Few-Shot)
# ============================================================================

EXAMPLES_NONE = ""

EXAMPLES_ONESHOT = (
    "Hier ist ein Beispiel:\n\n"
    "Beispiel-Transkript: "
    "'Ich schlafe nachts überhaupt nicht mehr. Morgens bin ich völlig ausgelaugt und kann mich nicht konzentrieren.'\n\n"
    "Beispiel-Output: "
    "[{'symptom': 'Insomnie', 'section': 'schlafe nachts überhaupt nicht mehr'}, "
    "{'symptom': 'Müdigkeit oder Energieverlust', 'section': 'völlig ausgelaugt'}]\n\n"
)

EXAMPLES_FEWSHOT = (
    "Hier sind mehrere Beispiele:\n\n"
    "Beispiel 1:\n"
    "Transkript: 'Seit zwei Wochen kann ich mich nicht mehr für meine Hobbys interessieren. "
    "Alles fühlt sich sinnlos an.'\n"
    "Output: [{'symptom': 'Interessen- oder Freudeverlust', 'section': 'kann ich mich nicht mehr für meine Hobbys interessieren'}, "
    "{'symptom': 'Schuld- oder Wertlosigkeitsgefühle', 'section': 'alles fühlt sich sinnlos an'}]\n\n"
    "Beispiel 2:\n"
    "Transkript: 'Meine Gedanken rasen, ich rede ständig und kann einfach nicht stillsitzen.'\n"
    "Output: [{'symptom': 'Ideenflucht oder Gedankenrasen', 'section': 'Gedanken rasen'}, "
    "{'symptom': 'Gesteigertes Sprechbedürfnis oder Rededruck', 'section': 'rede ständig'}, "
    "{'symptom': 'Gesteigerte zielgerichtete Aktivität oder psychomotorische Unruhe', 'section': 'kann nicht stillsitzen'}]\n\n"
)

EXAMPLES = {
    "zeroshot": EXAMPLES_NONE,
    "oneshot": EXAMPLES_ONESHOT,
    "fewshot": EXAMPLES_FEWSHOT,
}

# ============================================================================
# KOMPONENTE 6: Input-Text (Platzhalter)
# ============================================================================

INPUT_TRANSCRIPT = (
    "Hier ist das Interview-Transkript:\n\n"
    "{transcript}"
)

INPUT_FROM_PREVIOUS = (
    "Hier sind die Ergebnisse aus der vorherigen Task:\n\n"
    "{previous_results}"
)

INPUTS = {
    "transcript": INPUT_TRANSCRIPT,
    "previous_results": INPUT_FROM_PREVIOUS,
}

# ============================================================================
# TASK-DEFINITIONEN: Kombiniert alle 6 Bausteine
# ============================================================================

def build_prompt(task_id, role_id, example_type, input_type="transcript"):
    """
    Baut einen kompletten Prompt aus Bausteinen zusammen.
    
    Args:
        task_id: z.B. "task1_extract_symptoms"
        role_id: z.B. "psychiatrist" oder "none"
        example_type: z.B. "zeroshot", "oneshot", "fewshot"
        input_type: z.B. "transcript" oder "previous_results"
    
    Returns:
        Dict mit "system" und "user_template"
    """
    
    # 1. Role
    role = ROLES.get(role_id, "")
    
    # 2. Task
    task = TASKS.get(task_id, "")
    
    # 3. Format
    format_spec = FORMATS.get(task_id, "")
    
    # 4. Symptom-Liste (falls vorhanden)
    symptoms = SYMPTOM_LISTS.get(task_id, "")
    
    # 5. Beispiele
    examples = EXAMPLES.get(example_type, "")
    
    # 6. Input-Template (wird später mit Platzhaltern gefüllt)
    input_template = INPUTS.get(input_type, "")
    
    # Zusammensetzen zu System-Prompt
    system_parts = []
    if role:
        system_parts.append(role)
    system_parts.append(task)
    system_parts.append(format_spec)
    if symptoms:
        system_parts.append(symptoms)
    if examples:
        system_parts.append(examples)
    
    system_prompt = "\n\n".join(system_parts)
    
    return {
        "system": system_prompt,
        "user_template": input_template,
    }

# ============================================================================
# VORDEFINIERTE PROMPT-KONFIGURATIONEN
# ============================================================================

TASK_CONFIGS = {
    # TASK 1: Symptom-Extraktion
    "task1_zeroshot_no_role": build_prompt(
        "task1_extract_symptoms", "none", "zeroshot", "transcript"
    ),
    "task1_zeroshot_role": build_prompt(
        "task1_extract_symptoms", "role", "zeroshot", "transcript"
    ),
    "task1_oneshot_no_role": build_prompt(
        "task1_extract_symptoms", "none", "oneshot", "transcript"
    ),
    "task1_oneshot_role": build_prompt(
        "task1_extract_symptoms", "role", "oneshot", "transcript"
    ),
    "task1_fewshot_no_role": build_prompt(
        "task1_extract_symptoms", "none", "fewshot", "transcript"
    ),
    "task1_fewshot_role": build_prompt(
        "task1_extract_symptoms", "role", "fewshot", "transcript"
    ),
    
    # TASK 2: Diagnose
    "task2_zeroshot_no_role": build_prompt(
        "task2_diagnosis", "none", "zeroshot", "previous_results"
    ),
    "task2_zeroshot_role": build_prompt(
        "task2_diagnosis", "role", "zeroshot", "previous_results"
    ),
    
    # TASK 3: Severity
    "task3_zeroshot_no_role": build_prompt(
        "task3_severity", "none", "zeroshot", "previous_results"
    ),
    "task3_zeroshot_role": build_prompt(
        "task3_severity", "role", "zeroshot", "previous_results"
    ),
}

def get_prompt(template_id):
    """
    Hole eine vordefinierte Prompt-Konfiguration.
    
    Args:
        template_id: z.B. "task1_zeroshot_no_role"
    
    Returns:
        Dict mit "system" und "user_template"
    """
    return TASK_CONFIGS.get(template_id)
