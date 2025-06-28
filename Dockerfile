# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11-slim

# RUN apt-get update && apt-get install -y curl && \
#     curl -fsSL https://ollama.ai/install.sh | sh && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN useradd -m -u 1000 user
# USER user
# ENV HOME=/home/user \
#     PATH="/home/user/.local/bin:$PATH"

# # Création des répertoires nécessaires
# RUN mkdir -p $HOME/docker_ollama $HOME/logs
# WORKDIR $HOME/docker_ollama

# COPY --chown=user requirements.txt .
# RUN pip install --no-cache-dir --upgrade -r requirements.txt
# COPY --chown=user . .
# RUN chmod +x start.sh
# EXPOSE 7860 11434

# CMD ["./start.sh"]

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
