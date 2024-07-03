import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.embedding import generate_embeddings, preprocess_as_sentence_batches
from app.foundation import query_foundation_model, prepare_foundation_input, query_foundation_model_stream
from app.database import clean_db, init_db, store_embedding, search_embeddings


APP_PORT = int(os.getenv('APP_PORT', 8000))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    logger.info("Database initialized")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    text: str
    type: str  # "ingest", "query" or "query_stream"

class EmbeddingsRequest(BaseModel):
    text: str

class EmbeddingsResponse(BaseModel):
    embeddings: list

class ResponseModel(BaseModel):
    status: str
    response: str

# handles embeddings generation and persistence
def handle_ingest(text: str):
    sentence_batches = preprocess_as_sentence_batches(text)
    embedding_batches = generate_embeddings(sentence_batches)

    for i, (batch, embedding) in enumerate(zip(sentence_batches, embedding_batches)):
        batch_text = " ".join(batch)
        store_embedding(batch_text, embedding, 'web')

    return {"status": "success", "response": "embeddings stored successfully"}

# handles the retrieve/query operation
def handle_query(text: str):
    sentence_batches = preprocess_as_sentence_batches(text)
    embedding_batches = generate_embeddings(sentence_batches)

    search_results = search_embeddings(embedding_batches[0])
    enriched_context = " ".join([result[0] for result in search_results])
    input_prompt, token_count = prepare_foundation_input(enriched_context, text)
    response = query_foundation_model(input_prompt, token_count)

    return {"status": "success", "response": response}

# handles the retrieve/query operation as a stream
async def handle_query_stream(text: str):
    sentence_batches = preprocess_as_sentence_batches(text)
    embedding_batches = generate_embeddings(sentence_batches)

    search_results = search_embeddings(embedding_batches[0])
    enriched_context = " ".join([result[0] for result in search_results])
    input_prompt, token_count = prepare_foundation_input(enriched_context, text)
    return query_foundation_model_stream(input_prompt, token_count)

destination_handlers = {
    'ingest': handle_ingest,
    'query': handle_query,
    'query_stream': handle_query_stream
}

@app.post("/process", response_model=ResponseModel)
async def process_text(request: ProcessRequest):
    try:
        handler = destination_handlers.get(request.type)
        if handler:
            if request.type == 'query_stream':
                return StreamingResponse(await handler(request.text), media_type="text/plain")
            return handler(request.text)
        else:
            raise HTTPException(status_code=400, detail="Invalid destination specified")
    
    except Exception as e:
        logger.error(f"Error occurred during processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Something went wrong. It's not you. It's me.")


@app.post("/clean", response_model=ResponseModel)
async def clean_database():
    try:
        clean_db()
        return {"status": "success", "response": "database cleaned successfully"}
    except Exception as e:
        logger.error(f"Error occurred during database cleaning: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=APP_PORT, reload=True)
