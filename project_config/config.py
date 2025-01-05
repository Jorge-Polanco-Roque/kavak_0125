import os
import tempfile

from dotenv import load_dotenv, find_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv(find_dotenv())

# Variables de entorno requeridas
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWILIO_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_WHATSAPP_FROM = os.getenv('FROM')

# Mensaje de error predeterminado
ERROR_MESSAGE = (
    'Estamos teniendo algunos problemas con el servidor. '
    'Por favor, intenta de nuevo más tarde. :)'
)

# Configuración del directorio de salida
OUTPUT_DIR = os.path.join(
    tempfile.gettempdir(),
    'document-gpt',
    'output'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validar que todas las variables requeridas estén definidas
required_vars = ['OPENAI_API_KEY', 'TWILIO_TOKEN', 'TWILIO_SID', 'TWILIO_WHATSAPP_FROM']
for var in required_vars:
    if not locals()[var]:
        raise EnvironmentError(f"Missing required environment variable: {var}")
