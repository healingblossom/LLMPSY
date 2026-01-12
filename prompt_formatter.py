# prompt_formatter.py zum Formatieren der Prompts in das Format, dass das jeweilige Model braucht

def build_messages(system_text, user_text, model_config):
    """
    Baut Chat-Messages basierend auf Modell-Config.
    
    Args:
        system_text: System-Prompt
        user_text: User-Nachricht
        model_config: Dict mit Modell-Einstellungen (z.B. supports_system_role)
    
    Returns:
        List of dicts mit {'role': ..., 'content': ...}
    """
    
    # Standard: System und User getrennt
    if model_config.get("supports_system_role", True):
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
    else:
        # Fallback: System in User-Prompt zusammenführen
        merged = f"{system_text}\n\n{user_text}"
        return [{"role": "user", "content": merged}]
