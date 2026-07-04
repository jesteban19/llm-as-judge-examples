import pytest
import re
from repository_llm import invoke_draft_coach_agent

# =============================================
# Evals de Nivel 1 ejecutados contra el agente de Coach de Draft de Dota 2
# =============================================

@pytest.mark.asyncio
async def test_agent_recommendations_valid_offlaner():
    """
    Prueba que el agente recomiende un héroe viable para la Offlane (Posición 3) 
    y no un Hard Support o un héroe frágil.
    """
    # 1. EJECUCIÓN REAL: Le pedimos al agente un pick contra un Phantom Assassin.
    real_response = await invoke_draft_coach_agent(
        role="Offlane",
        enemy_carry="Phantom Assassin",
        intent="hero_recommendation"
    )

    print(f"Respuesta del Agente: {real_response}")
    
    # 2. EVALUACIÓN: Lista de Offlaners válidos y típicos (ground truth)
    valid_offlaners = ["axe", "centaur warrunner", "tidehunter", "bristleback", "slardar", "underlord", "dark seer","mars"]
    
    # Verificamos si al menos uno de los offlaners válidos está en la respuesta generada
    has_valid_pick = any(hero in real_response.lower() for hero in valid_offlaners)
    
    # Si el LLM recomienda "Crystal Maiden" o "Anti-Mage" para Offlane, el test falla.
    assert has_valid_pick, f"El agente falló en recomendar un héroe Offlane viable en el meta actual. Respuesta: {real_response}"

@pytest.mark.asyncio
async def test_agent_avoids_hallucinating_other_games():
    """
    Prueba que el LLM no alucine recomendando personajes de League of Legends u otros MOBAs.
    Esto es crucial porque los LLMs no están entrenados con ambos juegos.
    """
    real_response = await invoke_draft_coach_agent(
        role="Offlane",
        enemy_carry="Phantom Assassin",
        intent="hero_recommendation"
    )
    
    # Lista negra de términos que probarían una alucinación masiva de contexto
    banned_terms = ["garen", "darius", "league of legends", "campeón", "nexus"]
    
    has_hallucination = any(banned_word in real_response.lower() for banned_word in banned_terms)
    assert not has_hallucination, "Alerta de Alucinación: El agente usó terminología o personajes de otro juego."

@pytest.mark.asyncio
async def test_agent_includes_item_build_format():
    """
    Prueba que el agente respete el contrato de la interfaz y devuelva 
    una sugerencia de ítem clave usando el formato esperado.
    """
    real_response = await invoke_draft_coach_agent(
        role="Offlane",
        enemy_carry="Phantom Assassin",
        intent="hero_recommendation"
    )
    
    # El prompt del sistema exige que el bot siempre diga: "Ítem clave sugerido: [Nombre]"
    expected_format_regex = r"Ítem clave sugerido: \w+"
    matches = re.search(expected_format_regex, real_response, re.IGNORECASE)
    
    assert matches is not None, "El agente no incluyó la recomendación del ítem clave en el formato requerido."