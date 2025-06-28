#!/bin/bash
#MisterAI/Docker_Ollama
#start.sh_01
#https://huggingface.co/spaces/MisterAI/Docker_Ollama/


# Set environment variables for optimization HFSpace Free 2CPU - 16GbRAM
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1

# # Start Ollama in the background
# ollama serve &

# # Pull the model if not already present
# echo "gemma3:1b will be download"
# if ! ollama list | grep -q "gemma3:1b"; then
#     ollama pull gemma3:1b
# fi

# # Wait for Ollama to start up
# max_attempts=30
# attempt=0
# while ! curl -s http://localhost:11434/api/tags >/dev/null; do
#     sleep 1
#     attempt=$((attempt + 1))
#     if [ $attempt -eq $max_attempts ]; then
#         echo "Ollama failed to start within 30 seconds. Exiting."
#         exit 1
#     fi
# done

# echo "Ollama is Ready - gemma3:1b is Loaded"

python app.py