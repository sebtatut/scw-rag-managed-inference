import logging
import os
from typing import AsyncGenerator
import httpx
import requests
import json
import asyncio
from fastapi import HTTPException
from dotenv import load_dotenv

from app.tokenizer import tokenize_query_hf

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_SECRET = os.getenv('APP_SECRET')
API_ENDPOINT_PUB_FND = os.getenv('API_ENDPOINT_PUB_FND')

def prepare_foundation_input(context: str, user_query: str) -> tuple[list[dict], int]:
    # check params
    if not user_query:
        raise ValueError("Invalid input params")
    
    token_count = 0

    messages = [
        {"role": "system", "content": "You are ScaleBot, a factual and helpful assistant. You output Markdown messages."},
        {"role": "user", "content": f"Given this Context: {context}\nAnswer the question: {user_query}. Format your answer as Markdown."},
    ]

    for message in messages:
        token_count += len(tokenize_query_hf(message['content']))

    return messages, token_count


def query_foundation_model(messages: list[dict], token_count: int) -> str:
    # check params
    if not messages or token_count == 0:
        raise HTTPException(status_code=400, detail="Invalid input params")
    
    max_output_tokens = 4096 - token_count

    headers = {
        'Authorization': f'Bearer {APP_SECRET}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'meta/llama-3-8b-instruct:bf16',
        'messages': messages,
        'max_tokens': max_output_tokens,
        'temperature': 0.1,
        'stream': False
    }

    logger.info(f"Request data: {json.dumps(data)}")
    
    response = requests.post(API_ENDPOINT_PUB_FND, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Failed to get response from foundation model: {response.status_code}, {response.text}")

async def query_foundation_model_stream(messages: list[dict], token_count: int) -> AsyncGenerator[str, None]:
    # check params
    if not messages or token_count == 0:
        raise HTTPException(status_code=400, detail="Invalid input params")
    
    max_output_tokens = 8192 - token_count

    headers = {
        'Authorization': f'Bearer {APP_SECRET}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'meta/llama-3-8b-instruct:bf16',
        'messages': messages,
        'max_tokens': max_output_tokens,
        'temperature': 0.5,
        'stream': True
    }

    logger.info(f"Request data: {json.dumps(data)}")
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", API_ENDPOINT_PUB_FND, headers=headers, json=data) as response:
            if response.status_code != 200:
                logger.error(f"Failed to get response from foundation model: {response.status_code}, {response.text}")
                yield "Failed to get response from foundation model"
            
            async for line in response.aiter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    print(f"Line: {line}")
                    chunk_json = json.loads(line[6:])  # remove the "data: " prefix and parse the rest as JSON
                    # check if we have a message
                    if "choices" not in chunk_json or len(chunk_json["choices"]) == 0 or "delta" not in chunk_json["choices"][0]:                    
                        continue
                    chunk = chunk_json["choices"][0]["delta"]["content"]
                    yield chunk
