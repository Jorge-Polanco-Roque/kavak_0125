from openai import OpenAI
from project_config import config
from langchain_community.document_loaders.csv_loader import CSVLoader

import openai
import pandas as pd
from langchain_openai import OpenAI as LLMForAgent
from langchain_experimental.agents import create_pandas_dataframe_agent

# ========== 1) LLM PARA PREGUNTAS GENÉRICAS ==========
client = OpenAI(api_key=config.OPENAI_API_KEY)

def chat_completion(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[chat_completion] Error:", e)
        return config.ERROR_MESSAGE

# ========== 2) AGENTE DE PANDAS PARA PREGUNTAS SOBRE EL CSV ==========
openai.api_key = config.OPENAI_API_KEY

df = pd.read_csv(
    "/Users/A1064331/Desktop/pruebas/Kavak/test_1/input/sample_caso_ai_engineer.csv",
    delimiter=","
)
agent = create_pandas_dataframe_agent(
    llm=LLMForAgent(openai_api_key=openai.api_key, temperature=0.0),
    df=df,
    verbose=False,
    allow_dangerous_code=True
)

# ========== 3) DETECCIÓN DE PALABRAS CLAVE PARA CSV ==========
CSV_KEYWORDS = [
    "stock_id","km","price","make","model","year",
    "version","bluetooth","largo","ancho","altura","car_play"
]

def needs_csv(query: str) -> bool:
    """Retorna True si la pregunta menciona columnas del CSV."""
    return any(k in query.lower() for k in CSV_KEYWORDS)

# ========== 4) FUNCIÓN CENTRAL PARA DECIDIR ENTRE AGENTE O LLM ==========
def ask_question(query: str) -> str:
    """
    - Si la pregunta menciona datos del CSV, usa el agente de Pandas.
      El agente devuelve un diccionario con 'input' y 'output'.
    - Si NO requiere CSV, llama a 'chat_completion'.
    """
    if needs_csv(query):
        try:
            response = agent.invoke(query)
            # El agente retorna algo como:
            # {'input': 'pregunta...', 'output': 'respuesta...'}
            return response.get("output", "")
        except Exception as e:
            print("[Agent] Error:", e)
            return config.ERROR_MESSAGE
    else:
        return chat_completion(query)

# ========== 5) FUNCIÓN DE PRUEBAS ==========
def testing_function(query: str) -> None:
    """
    Imprime la pregunta y la respuesta,
    evitando retornar None y mostrando sólo 'output'.
    """
    print(f"Pregunta: {query}")
    answer = ask_question(query)
    print("Respuesta:", answer)

# Llamadas de prueba (si quieres que se ejecuten al importar/quitar None):
#testing_function("¿Cuándo fue la independencia de México?")
testing_function("¿Cuál es el precio promedio de los autos con bluetooth habilitado?")
