from openai import OpenAI
from project_config import config

from langchain_community.document_loaders.csv_loader import CSVLoader


client = OpenAI(
    api_key= config.OPENAI_API_KEY,
)

def chat_completion(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
            )
        return response.choices[0].message.content.strip()
    except:
        return config.ERROR_MESSAGE

loader = CSVLoader(file_path='/Users/A1064331/Desktop/pruebas/Kavak/test_1/input/sample_caso_ai_engineer.csv',
                   csv_args={
                       "delimiter": ",",
                       "fieldnames": ["stock_id","km","price","make","model",
                                      "year","version","bluetooth","largo","ancho,altura","car_play"]})

data = loader.load()

prompt_test = """
Esta es una prueba muy complicada, qué opinas al respecto?
"""

#print(chat_completion(prompt_test))





import openai
import pandas as pd
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1) Configura tu API key de OpenAI
openai.api_key = config.OPENAI_API_KEY

# 2) Carga tu CSV en un DataFrame de pandas
df = pd.read_csv(
    "/Users/A1064331/Desktop/pruebas/Kavak/test_1/input/sample_caso_ai_engineer.csv",  # Ruta a tu CSV
    delimiter=",",
    # Si no tiene encabezados en el CSV, podrías usar 'names=[...]'
)

# 3) Crea un agente con el DataFrame
#    - El agente usará internamente pandas para hacer cálculos reales.
#    - 'OpenAI' proviene de 'langchain_openai' y se encarga de la parte LLM.
agent = create_pandas_dataframe_agent(
    llm=OpenAI(openai_api_key=openai.api_key, temperature=0.0),
    df=df,
    verbose=False,
    allow_dangerous_code=True
)

# 4) Realiza consultas en lenguaje natural
#    Con este Agent, las preguntas que requieran cálculos sobre las columnas
#    se responden ejecutando código pandas "bajo el capó".
if __name__ == "__main__":
    # Ejemplo: precio promedio de autos con bluetooth habilitado
    query_1 = "¿Cuál es el precio promedio de los autos con bluetooth habilitado?"
    result_1 = agent.invoke(query_1)
    print(f"\nPregunta: {query_1}\nRespuesta: {result_1}")

    # Ejemplo: cuántos autos tienen CarPlay
    query_2 = "¿Cuántos autos tienen CarPlay?"
    result_2 = agent.invoke(query_2)
    print(f"\nPregunta: {query_2}\nRespuesta: {result_2}")

    # Agrega más consultas según necesites...

