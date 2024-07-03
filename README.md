# scw-rag-managed-inference
Basic RAG pipeline using Scaleway's Managed Inference and PostgreSQL Managed Database with pgvector extension

# Prerequisites

1. A Scaleway Account with IAM privileges so you can create API Keys and Service Accounts
2. A Managed Inference deployment hosting a llama-3-8b-instruct foundation model
3. A Managed Inference deployment hosting a sentence-t5-xxl embeddings model
4. A Managed PostgreSQL Database
5. A Huggingface token for using the Transformers library

# Environment Variables

You'll need to generate an API Key for the Ingest, Processing Service as well as the Managed Inference deployments. Make sure that the API Key for the Inference Service also has the Object Storage preferred Project option checked and setup.

The Ingest Service uses the following environment variables: 

* `APP_PORT`: the port on which the app is exposed 
* `APP_KEY_ID`: the ID of the API Key 
* `APP_SECRET`: the secret of the API Key

The Processing Service uses the following environment variables: 

* `APP_PORT`
* `APP_KEY_ID`
* `APP_SECRET`
* `API_ENDPOINT_PUB_EMB`
* `API_ENDPOINT_PUB_FND`
* `API_KEY_ID`
* `API_SECRET`
* `POSTGRES_DB`
* `POSTGRES_USER`
* `POSTGRES_PASSWORD`
* `POSTGRES_HOST`
* `POSTGRES_PORT`
* `HUGGINGFACE_TOKEN`