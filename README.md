<div align="center">
  <img src="https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=1200&auto=format&fit=crop" alt="AI Evaluation Banner" width="100%" style="border-radius: 10px; margin-bottom: 20px;">

  <h1>🧪 LLM-as-a-Judge Examples</h1>
  <p><strong>Evaluación Sistemática y Pruebas Unitarias para Agentes de IA</strong></p>

  <!-- Badges -->
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://docs.pytest.org/en/stable/">
    <img src="https://img.shields.io/badge/Pytest-Asyncio-4A4A4A.svg?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest">
  </a>
  <a href="https://github.com/microsoft/agent-framework">
    <img src="https://img.shields.io/badge/Microsoft-Agent_Framework-0078D4.svg?style=for-the-badge&logo=microsoft&logoColor=white" alt="Agent Framework">
  </a>
  <a href="https://openai.com/">
    <img src="https://img.shields.io/badge/OpenAI-GPT_4o-412991.svg?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  </a>
</div>

---

## 📖 Descripción General

Este repositorio contiene patrones de arquitectura, ejemplos y configuraciones para implementar **Sistemas de Evaluación (Evals)** robustos en aplicaciones impulsadas por Modelos de Lenguaje Grande (LLMs).

Utilizando el **Microsoft Agent Framework** y **pytest**, este proyecto demuestra cómo transicionar de simples "vibe checks" a pruebas determinísticas y automatizadas (Nivel 1 y Nivel 2 de Evals) para garantizar la calidad, seguridad y precisión de los agentes inteligentes.

## ✨ Características Principales

- **🛡️ Guardrails de Seguridad:** Pruebas automatizadas para evitar fugas de información sensible (PII) mediante expresiones regulares y validación de salidas.
- **🤖 LLM-as-a-Judge:** Implementación de aserciones asíncronas para evaluar la calidad semántica y el enrutamiento de agentes.
- **🎮 Casos de Uso Reales:** Incluye ejemplos prácticos, como un agente _Coach de Drafts de Dota 2_, validando el cumplimiento del meta-juego y previniendo alucinaciones inter-dominio.
- **⚡ Integración Continua (CI):** Configurado para ejecutarse sin fricciones en pipelines, evaluando el impacto de cada modificación en el _System Prompt_.

---

## 🛠️ Stack Tecnológico

| Componente       | Tecnología / Librería      | Propósito                                            |
| :--------------- | :------------------------- | :--------------------------------------------------- |
| **Orquestación** | `agent-framework`          | Construcción y gestión del ciclo de vida del agente. |
| **Testing**      | `pytest`, `pytest-asyncio` | Arnés de pruebas y aserciones asíncronas.            |
| **Modelos**      | `OpenAI API`               | Motor de inferencia principal (gpt-4o).              |
| **Entorno**      | `python-dotenv`            | Gestión segura de variables de entorno y secretos.   |

---

## 🚀 Requisitos Previos

Antes de clonar el proyecto, asegúrate de tener instalado:

- Python 3.12 o superior.
- Una cuenta activa de OpenAI con créditos para la API.
- Git.

---

## 💻 Instalación y Configuración

**1. Clonar el repositorio:**

```bash
git clone [https://github.com/jesteban19/llm-as-judge-examples.git](https://github.com/jesteban19/llm-as-judge-examples.git)
cd llm-as-judge-examples
```
