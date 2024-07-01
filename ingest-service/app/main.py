import os
import logging
import uuid
from typing import List
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.scraper import scrape_web_page, parse_pdf_from_s3
import boto3
import httpx
import asyncio

load_dotenv()

APP_PORT = int(os.getenv('APP_PORT', 8080))
APP_KEY_ID = os.getenv('APP_KEY_ID')
APP_SECRET = os.getenv('APP_SECRET')
PROCESSING_SERVICE_URL = os.getenv('PROCESSING_SERVICE_URL', 'http://localhost:8000/process')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3', 
                         endpoint_url='https://s3.fr-par.scw.cloud',
                         aws_access_key_id=APP_KEY_ID,
                         aws_secret_access_key=APP_SECRET)

operation_status = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # check if s3_client is properly configured
    try:
        s3_client.list_buckets()
    except Exception as e:
        logger.error(f"Error occurred during S3 client configuration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    yield
    logger.info("Object Storage OK")

app = FastAPI(lifespan=lifespan)

class ScrapeRequest(BaseModel):
    urls: List[str]

class ParseRequest(BaseModel):
    bucket_name: str
    pdf_keys: List[str]

class ResponseModel(BaseModel):
    status: str
    operation_id: str

class OperationStatusResponse(BaseModel):
    status: str


async def send_to_processing_service(text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            PROCESSING_SERVICE_URL,
            json={"text": text, "type": "ingest"}
        )
        response.raise_for_status()


async def process_scrape_task(urls: List[str], operation_id: str):
    try:
        tasks = []
        for url in urls:
            text = scrape_web_page(url)
            tasks.append(send_to_processing_service(text))
        await asyncio.gather(*tasks)
        operation_status[operation_id] = "completed"
    except Exception as e:
        logger.error(f"Error occurred during scraping: {str(e)}", exc_info=True)
        operation_status[operation_id] = f"failed: {str(e)}"


async def process_parse_task(bucket_name: str, pdf_keys: List[str], operation_id: str):
    try:
        tasks = []
        for pdf_key in pdf_keys:
            text = parse_pdf_from_s3(bucket_name, pdf_key, s3_client)
            tasks.append(send_to_processing_service(text))
        await asyncio.gather(*tasks)
        operation_status[operation_id] = "completed"
    except Exception as e:
        logger.error(f"Error occurred during PDF parsing: {str(e)}", exc_info=True)
        operation_status[operation_id] = f"failed: {str(e)}"


@app.post("/scrape", response_model=ResponseModel)
async def scrape_web(request: ScrapeRequest, background_tasks: BackgroundTasks):
    try:
        operation_id = str(uuid.uuid4())
        operation_status[operation_id] = "in_progress"
        background_tasks.add_task(process_scrape_task, request.urls, operation_id)
        return {"status": "started", "operation_id": operation_id}
    except Exception as e:
        logger.error(f"Error occurred during scraping: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse", response_model=ResponseModel)
async def parse_pdf(request: ParseRequest, background_tasks: BackgroundTasks):
    try:
        operation_id = str(uuid.uuid4())
        operation_status[operation_id] = "in_progress"
        background_tasks.add_task(process_parse_task, request.bucket_name, request.pdf_keys, operation_id)
        return {"status": "started", "operation_id": operation_id}
    except Exception as e:
        logger.error(f"Error occurred during PDF parsing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{operation_id}", response_model=OperationStatusResponse)
async def get_status(operation_id: str):
    status = operation_status.get(operation_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Operation ID not found")
    return {"status": status}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=APP_PORT, reload=True)
