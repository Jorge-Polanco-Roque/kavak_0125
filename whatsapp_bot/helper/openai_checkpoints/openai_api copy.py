import os
import uuid

from openai import OpenAI
import requests
#import soundfile as sf

from project_config import config

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
    
