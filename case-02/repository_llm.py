import os
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
import dotenv

dotenv.load_dotenv("../.env")

async def invoke_draft_coach_agent(role: str, enemy_carry: str, intent: str) -> str:
    """
    Orquesta la llamada al LLM utilizando Microsoft Agent Framework 
    para actuar como un coach de Draft de Dota 2.
    """
    # 1. Diseñar el System Prompt (El núcleo de tu Agente)
    # Aquí definimos las reglas exactas que nuestros Unit Tests van a evaluar
    instructions = """
    Eres un coach profesional y analista de Dota 2 de alto MMR especializado en la fase de draft.
    Tu objetivo es analizar los picks del equipo enemigo y recomendar un héroe viable para tu equipo.
    
    REGLAS ESTRICTAS QUE DEBES CUMPLIR:
    1. CONTEXTO DE JUEGO: Utiliza ÚNICAMENTE terminología de Dota 2 (son "héroes", no "campeones"). 
       Bajo ninguna circunstancia menciones personajes de League of Legends u otros juegos.
    2. VIABILIDAD: Recomienda héroes que sean meta y tradicionalmente viables para la posición solicitada. 
    3. FORMATO DE SALIDA: Debes explicar brevemente tu recomendación y, obligatoriamente, terminar 
       tu respuesta con la sugerencia de un ítem para pelear contra el carry enemigo usando EXACTAMENTE este formato:
       Ítem clave sugerido: [Nombre del Ítem]
    """

    # 2. Inicializar el Cliente del Modelo
    # Agent Framework resuelve automáticamente variables como OPENAI_API_KEY desde el entorno.
    # (Si usas Azure o Foundry, simplemente cambias esto por AzureOpenAIChatClient o FoundryChatClient)
    client = OpenAIChatClient(
        api_key=os.environ.get("TOKEN"),
        base_url=os.environ.get("ENDPOINT"),
        model="gpt-4.1"
        #temperature=0.3  # Mantenemos precisión analítica para el draft
    )
    
    # 3. Construir el Agente
    agent = Agent(
        client=client,
        instructions=instructions
    )

    # 4. Formatear la solicitud del usuario
    user_message = (
        f"El equipo enemigo acaba de seleccionar a {enemy_carry} como su Hard Carry. "
        f"Necesito una recomendación táctica para jugar en la posición de {role} "
        f"y ganar mi línea. Mi intención principal es: {intent}."
    )

    # 5. Ejecutar el agente y retornar el resultado a nuestros tests
    result = await agent.run(user_message)
    
    return str(result)