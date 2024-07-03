import os
import logging
from dotenv import load_dotenv
import requests
import json

from app.tokenizer import count_tokens_embedding, tokenize_data_hf, tokenize_query_hf, tokenize_sentences, tokenize_words

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_ENDPOINT_PUB_EMB = os.getenv('API_ENDPOINT_PUB_EMB')
APP_SECRET = os.getenv('APP_SECRET')
MAX_TOKENS_EMBEDDINGS = 64

# preprocess text into batches of sentences
def preprocess_as_sentence_batches(text: str) -> list[list[str]]:
    sents_and_token_counts = tokenize_sentences(text)
    batches = []
    current_batch = []

    current_length = 0
    for sentence, token_count in sents_and_token_counts:
        if token_count > MAX_TOKENS_EMBEDDINGS:
            print(f"BIG sentence: {sentence}, tokens: {token_count}")


        if current_length + token_count > MAX_TOKENS_EMBEDDINGS:
            batches.append(current_batch)
            current_batch = []
            current_length = 0
        
        current_batch.append(sentence)
        current_length += token_count

    if current_batch:
        batches.append(current_batch)

    return batches

# preprocess text into batches of words
def preprocess_as_words(text: str) -> list[list[str]]:
    tokens = tokenize_words(text)
    token_count = len(tokens)

    batches = []
    current_batch = []

    if token_count < MAX_TOKENS_EMBEDDINGS:
        batches.append(tokens)
        return batches

    current_length = 0
    for token in tokens:
        if current_length + 1 > MAX_TOKENS_EMBEDDINGS:
            batches.append(current_batch)
            current_batch = []
            current_length = 0
        
        current_batch.append(token)
        current_length += 1

    if current_batch:
        batches.append(current_batch)

    return batches

# preprocess text into batches of subwords
def preprocess_as_subwords(text: str) -> list[list[str]]:
    tokens = tokenize_data_hf(text)
    token_count = len(tokens)

    batches = []
    current_batch = []

    if token_count < MAX_TOKENS_EMBEDDINGS:
        batches.append(tokens)
        return batches

    current_length = 0
    for token in tokens:
        if current_length + 1 > MAX_TOKENS_EMBEDDINGS:
            batches.append(current_batch)
            current_batch = []
            current_length = 0
        
        current_batch.append(token)
        current_length += 1

    if current_batch:
        batches.append(current_batch)

    return batches

# generate embeddings batches for text batches
def generate_embeddings(sentence_batches: list[list[str]]) -> list[list[float]]:
    embeddings_batches = []
    
    headers = {
        'Authorization': f'Bearer {APP_SECRET}',
        'Content-Type': 'application/json'
    }

    logger.info(f"Total batches: {len(embeddings_batches)}")
    
    for batch in sentence_batches:
        data = {
            'input': batch,
            'model': 'sentence-transformers/sentence-t5-xxl:fp32'
        }

        num_tokens = count_tokens_embedding(batch)

        logger.info(f"Requesting embeddings for {num_tokens} tokens")
        
        response = requests.post(API_ENDPOINT_PUB_EMB, headers=headers, data=json.dumps(data))
        
        if response.status_code != 200:
            logger.error(f"Failed to get embeddings for batch: {batch}, tokens: {num_tokens}")
            continue

        batch_embeddings = response.json()['data'][0]['embedding']
        embeddings_batches.append(batch_embeddings)
    
    return embeddings_batches
