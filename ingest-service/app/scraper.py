import re
from bs4 import BeautifulSoup
import requests
import PyPDF2
from io import BytesIO


# scrape web page and return raw text
def scrape_web_page(url):
    # check params
    if len(url) == 0:
        raise ValueError("URL should not be empty")

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    raw_text = soup.get_text()
    return raw_text


# parse PDF from S3 and return raw text
def parse_pdf_from_s3(bucket_name, pdf_key, s3_client):
    # check params
    if len(bucket_name) == 0:
        raise ValueError("Bucket name should not be empty")
    
    if len(pdf_key) == 0:
        raise ValueError("PDF key should not be empty")
    
    if s3_client is None:
        raise ValueError("S3 client should not be None")

    pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
    pdf_content = pdf_obj['Body'].read()
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
