# Usa imagem Python oficial
FROM python:3.10-slim

# evitar buffering (melhor logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copia ficheiros de dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código
COPY . .

# Port que o container vai escutar (padrão comum: 8080)
ENV PORT=7860

# Comando para iniciar a FastAPI com uvicorn
CMD ["sh", "-c", "uvicorn FlashClassifier.api:app --host 0.0.0.0 --port $PORT"]