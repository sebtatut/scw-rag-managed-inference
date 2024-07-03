import os
import re
from transformers import AutoTokenizer
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from dotenv import load_dotenv

load_dotenv()

nlp = spacy.load('en_core_web_sm')


HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# for meta/llama-3-8b-instruct:bf16
ID_MODEL_FOUNDATION = 'meta-llama/Meta-Llama-3-8B-Instruct'
# for sentence-transformers/sentence-t5-xxl:fp32
ID_MODEL_EMBEDDING = 'sentence-transformers/sentence-t5-xxl'

# load tokenizer for meta/llama-3-8b-instruct:bf16
tokenizer_foundation = AutoTokenizer.from_pretrained(ID_MODEL_FOUNDATION, token=HUGGINGFACE_TOKEN)

# load tokenizer for sentence-transformers/sentence-t5-xxl:fp32
tokenizer_embedding = AutoTokenizer.from_pretrained(ID_MODEL_EMBEDDING, token=HUGGINGFACE_TOKEN)


section_patterns = [
    r'^[0-9]+\.$',        # Matches "1.", "2.", "3.", etc.
    r'^[A-Z]\.$',         # Matches "A.", "B.", "C.", etc.
    r'^[a-z]\)\.$',       # Matches "a).", "b).", "c).", etc.
    r'^[A-Z]\.[0-9]+$',   # Matches "A.2", "B.3", etc.
    r'^[0-9]+\.[0-9]+$',  # Matches "1.1", "2.2", etc.
]

def is_section_pattern(text: str) -> bool:
    return any(re.match(pattern, text) for pattern in section_patterns)

def normalize_text(text: str) -> str:
    # remove newlines and carriage returns
    text = re.sub(r'[\r\n]+', ' ', text)
    # replacxe special characters with space
    text = re.sub(r'[^\w\s]', ' ', text)
    # remove special characters + punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # convert to lowercase
    text = text.lower()

    return text

# normalize text
def normalize_data(text: str) -> str:
    normalized = tokenizer_embedding.backend_tokenizer.normalizer.normalize_str(text)
    # remove extra spaces
    normalized = ' '.join(normalized.split())

    return normalized

# tokenize user query
def tokenize_query_hf(text: str) -> list[str]:
    return tokenizer_foundation.tokenize(text)

# tokenize data for embeddings generation
def tokenize_data_hf(text: str) -> list[str]:
    return tokenizer_embedding.tokenize(text)

# tokenize to sentences
def tokenize_sentences(text: str) -> list[tuple[str, int]]:
    doc = nlp(text)

    single_words = []
    sentences = []
    for sent in doc.sents:
        sentence = sent.text
        if len(sentence) < 3:
            print(f"Problem sentence: {sentence}")
            
            if is_section_pattern(sentence): 
                print("Section pattern")

            single_words.append(sentence)
            continue
        
        if len(single_words) != 0:
            # add single words to the sentence
            sentence = " ".join(single_words) + " " + sentence
            single_words = []
            print(f"With added single words: {sentence}")

        normalized = normalize_data(sentence)
        normalized = normalize_text(normalized)

        # get size in tokens
        token_count = len(tokenize_data_hf(sentence))

        sent_and_token_count = (normalized, token_count)

        sentences.append(sent_and_token_count)

    return sentences

# tokenize to words
def tokenize_words(text: str) -> list[str]:
    doc = nlp(text)
    words = [token.text for token in doc]
    return words

def count_tokens_embedding(text: list) -> int:
    num_tokens = 0

    for t in text:
        num_tokens += len(tokenize_data_hf(t))

    return num_tokens

def count_tokens_foundation(text: list) -> int:
    num_tokens = 0

    for t in text:
        num_tokens += len(tokenize_query_hf(t))

    return num_tokens
