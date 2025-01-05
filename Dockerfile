# Usa una imagen oficial de Python
FROM python:3.9-slim

# Crea el directorio de la aplicación
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt requirements.txt

# Instala las dependencias (se desactiva la caché para reducir el tamaño)
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo tu proyecto al contenedor
COPY . .

# Expone el puerto que usa Flask (ajusta si cambias en tu app)
EXPOSE 5000

# Comando para ejecutar la app. 
# Ajusta la ruta según dónde esté tu main.py
CMD ["python", "whatsapp_bot/src/main.py"]
