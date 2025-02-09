import io
import base64
from typing import Dict
import os
import runpod
from huggingface_hub import login, snapshot_download
from transformers import pipeline
import torch

HUGGING_FACE_TOKEN = ''

login(HUGGING_FACE_TOKEN)

MODEL_DIR = "/tmp/mistral_7b_instruct"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading Mistral-7B-Instruct-v0.3 model...")
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir=MODEL_DIR,
    token=HUGGING_FACE_TOKEN,  
    ignore_patterns=["*.md", "*.txt", "*.pdf"] 
)


chatbot = pipeline(
    "text-generation",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Model loaded successfully!")


def handler(job: Dict[str, any]) -> str:
    """
    Handler function for processing a job.

    Args:
        job (dict): A dictionary containing the job input.

    Returns:
        str: The generated Jeep-themed chatbot response.
    """

    job_input = job.get("input", {})
    input_text = job_input.get("text") or job_input.get("prompt")

    if not input_text or not isinstance(input_text, str):
        return "Error: No valid text input provided."

    messages = [
        {"role": "system", "content": "You are a Jeep enthusiast chatbot. Every response you give must reference Jeep in some way."},
        {"role": "user", "content": input_text}
    ]

    response = chatbot(messages, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)

    return response[0]["generated_text"]


runpod.serverless.start({"handler": handler})
