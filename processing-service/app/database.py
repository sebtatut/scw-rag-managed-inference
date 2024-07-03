import logging
import os
from dotenv import load_dotenv
import psycopg2
from nltk.corpus import wordnet
import numpy as np

from app.tokenizer import tokenize_query_hf

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")


def get_connection():
    return psycopg2.connect(
        dbname = POSTGRES_DB,
        user = POSTGRES_USER,
        password = POSTGRES_PASSWORD,
        host = POSTGRES_HOST,
        port = POSTGRES_PORT
    )


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id BIGSERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding VECTOR(768) NOT NULL,
        doc_type TEXT NOT NULL
    );
    """)
    # create idx on the embedding column to optimize NN searches
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_embedding ON documents USING hnsw (embedding vector_cosine_ops);
    """)

    conn.commit()
    cur.close()
    conn.close()


def clean_db():
    conn = get_connection()
    cur = conn.cursor()

    delete_query = "DELETE FROM documents"
    
    cur.execute(delete_query)

    conn.commit()
    cur.close()
    conn.close()


def store_embedding(text: str, embedding, doc_type):
    # check params
    if len(embedding) != 768:
        raise ValueError("Embedding should have 768 dimensions") # see https://huggingface.co/sentence-transformers/sentence-t5-xxl
    
    if len(text) == 0:
        raise ValueError("Text should not be empty")
    
    if len(doc_type) == 0:
        raise ValueError("Doc type should not be empty")

    conn = get_connection()
    cur = conn.cursor()

    # already exists?
    query = """
    SELECT 1 FROM documents WHERE text = %s AND doc_type = %s;
    """
    cur.execute(query, (text, doc_type))
    result = cur.fetchone()

    if result is None:
        insert_query = """
        INSERT INTO documents (text, embedding, doc_type) VALUES (%s, %s, %s);
        """
        cur.execute(insert_query, (text, embedding, doc_type))
        conn.commit()

    cur.close()
    conn.close()

def search_embeddings(query_embedding):
    # check params
    if len(query_embedding) != 768:
        raise ValueError("Embedding should have 768 dimensions") # see https://huggingface.co/sentence-transformers/sentence-t5-xxl

    conn = get_connection()
    cur = conn.cursor()

    search_query_1 = """
    WITH cte AS (
    SELECT text, (embedding <#> %s::vector(768)) as similarity 
    FROM documents
    ORDER BY similarity asc
    LIMIT 10
    )
    SELECT * FROM cte
    WHERE similarity < -0.75
    """

    search_query = """
    SELECT text FROM documents
    ORDER BY embedding <=> %s::vector(768)
    LIMIT 10
    """

    cur.execute(search_query, (query_embedding,))
    results = cur.fetchall()

    selected_results = []
    token_count = 0

    for result in results:
        tokens = len(tokenize_query_hf(result[0]))

        if token_count + tokens > 7168: # 8192 - 1024
            break

        selected_results.append(result)
        token_count += tokens

    cur.close()
    conn.close()
    return results
