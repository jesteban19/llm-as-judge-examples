import pytest
import json
from pydantic import BaseModel, Field, ValidationError
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
import dotenv
import os
import re

dotenv.load_dotenv("../.env")

# ==========================================
# 1. Definir el Esquema de Salida Estructurada (Pydantic)
# ==========================================
class EvalScore(BaseModel):
    score: int = Field(description="Puntuación entera del 1 al 5.", ge=1, le=5)
    reasoning: str = Field(description="Explicación detallada de por qué se otorgó esa puntuación.")

# ==========================================
# 2. Construir el Agente Juez (LLM-as-a-Judge)
# ==========================================
async def evaluate_with_llm_judge(question: str, expected_context: str, actual_response: str) -> EvalScore:
    """
    Instancia un agente diseñado exclusivamente para evaluar respuestas de otros agentes.
    """
    client = OpenAIChatClient(
        model="gpt-4.1",
        base_url=os.environ.get("ENDPOINT"),
        api_key=os.environ.get("TOKEN")
    )
    
    # El prompt del juez con la rúbrica del 1 al 5
    instructions = """
    Eres un evaluador experto de sistemas de IA. Tu tarea es calificar la respuesta generada 
    por un agente del 1 al 5, comparándola con el contexto esperado o la respuesta ideal.
    
    RÚBRICA DE PUNTUACIÓN:
    5 - Excelente: La respuesta es precisa, completa, respeta el contexto y es altamente útil.
    4 - Buena: La respuesta es correcta pero omite un detalle menor o el formato es ligeramente imperfecto.
    3 - Aceptable: La respuesta aborda la pregunta pero tiene imprecisiones leves o falta de profundidad.
    2 - Pobre: La respuesta contiene errores significativos, ignora restricciones o da información engañosa.
    1 - Fallo Crítico: La respuesta es totalmente incorrecta, irrelevante, o alucina de forma grave.
    
    Analiza la respuesta paso a paso y devuelve un JSON estricto con el puntaje ('score') y tu justificación ('reasoning').
    Devuelve un JSON con este formato:
    {
        "score": <entero>,
        "reasoning": "<texto>"
    }
    """
    
    judge_agent = Agent(
        client=client,
        instructions=instructions
    )
    
    # Construimos el caso a evaluar
    evaluation_prompt = (
        f"Pregunta del Usuario: {question}\n"
        f"Contexto/Respuesta Ideal: {expected_context}\n"
        f"Respuesta Generada por el Agente: {actual_response}\n"
    )
    
    # Ejecutamos el juez
    raw_result = str(await judge_agent.run(evaluation_prompt))
    # 2. Parseamos limpiamente con nuestra función
    try:
        evaluation: EvalScore = parse_pydantic_from_llm(raw_result, EvalScore)
        return evaluation
    except (ValueError, ValidationError) as e:
        # Fallback de seguridad si el modelo alucina totalmente
        return EvalScore(
            score=1, 
            reasoning=f"Fallo de Parseo: {e}"
        )

def parse_pydantic_from_llm(raw_text: str, model_class: type[BaseModel]) -> BaseModel:
    """
    Busca un bloque JSON válido dentro de una respuesta de texto libre del LLM
    y lo convierte en una instancia del modelo Pydantic proporcionado.
    """
    # Expresión regular que busca todo lo que esté entre el primer '{' y el último '}'
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    
    if not match:
        raise ValueError(f"No se detectó ninguna estructura JSON en el output del LLM. Texto recibido: {raw_text}")
    
    json_string = match.group(0)
    
    # Pydantic se encarga de validar que las llaves y tipos de datos sean correctos
    return model_class.model_validate_json(json_string)
# ==========================================
# 3. El Dataset de Evaluación (Golden Dataset)
# ==========================================
# En la vida real, podrías cargar esto desde un archivo JSONL o CSV
EVALUATION_DATASET = [
    {
        "id": "caso_exitoso",
        "question": "Necesito un Offlaner contra Phantom Assassin.",
        "expected_context": "Debe recomendar un tanque con armadura o daño de retorno, como Axe o Centaur, usando solo términos de Dota 2.",
        "actual_response": "Te recomiendo jugar Axe en la Offlane. Con tu Llamada del Berserker ignoras su evasión y la obligas a atacarte. Ítem clave sugerido: Malla de Cuchillas."
    },
    {
        "id": "caso_alucinacion_lol",
        "question": "Necesito un Offlaner contra Phantom Assassin.",
        "expected_context": "Debe recomendar un tanque con armadura o daño de retorno, como Axe o Centaur, usando solo términos de Dota 2.",
        "actual_response": "Te recomiendo jugar con Garen en la top lane. Es un campeón muy tanque. Ítem clave sugerido: Capa de Fuego."
    },
    {
        "id": "caso_respuesta_incompleta",
        "question": "Necesito un Offlaner contra Phantom Assassin.",
        "expected_context": "Debe recomendar un tanque con armadura o daño de retorno, como Axe o Centaur, usando solo términos de Dota 2.",
        "actual_response": "Axe es bueno." # Respuesta pobre, no tiene el formato ni explica.
    }
]

# ==========================================
# 4. Las Pruebas Automatizadas con Pytest
# ==========================================
@pytest.mark.asyncio
@pytest.mark.parametrize("record", EVALUATION_DATASET, ids=lambda x: x["id"])
async def test_llm_response_quality(record):
    """
    Itera sobre el dataset y utiliza el LLM-as-a-judge para evaluar cada registro.
    """
    # 1. Llamamos al juez
    evaluation: EvalScore = await evaluate_with_llm_judge(
        question=record["question"],
        expected_context=record["expected_context"],
        actual_response=record["actual_response"]
    )
    
    # 2. Imprimimos el razonamiento del juez para tener visibilidad si la prueba falla
    print(f"\n--- Razón del Juez: {evaluation.reasoning} ---")
    
    # 3. Aserción de Calidad: Exigimos que la respuesta tenga al menos un 4 sobre 5
    assert evaluation.score >= 4, (
        f"Prueba fallida. El agente generó una respuesta de baja calidad (Puntaje: {evaluation.score}).\n"
        f"Justificación del Juez: {evaluation.reasoning}"
    )