import re
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import docx2txt
import logging
from typing import Dict, List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pandas as pd
from src.openai_helpers import generate_description, generate_keywords
from src.redis_helpers import cache_value, redis_client, get_cached_value
from PyPDF2 import PdfReader

load_dotenv()

document_types = ['pdf', 'docx', 'xlsx', 'jpg', 'png']
document_categories = ['bank_statement', 'drivers_license', 'invoice'] 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def generate_class_descriptions(
    class_name: str,
    num_descriptions: int = 2,
    model: str = "gpt-4",
    max_tokens: int = 100,
    temperature: float = 0.7
) -> List[str]:
    redis_key = f"class_descriptions:{class_name}"
    descriptions = get_cached_value(redis_client, redis_key)

    prompt_template = (
        "Provide a detailed and descriptive explanation of what a '{class_name}' document is. "
        "Include key characteristics and typical contents."
    )

    missing_count = num_descriptions - len(descriptions)
    if missing_count > 0:
        logging.info(f"Generating {missing_count} descriptions for '{class_name}'...")
        for _ in range(missing_count):
            prompt = prompt_template.format(class_name=class_name.replace('_', ' '))
            description = generate_description(prompt, model, max_tokens, temperature)
            if description:
                descriptions.append(description)

    return descriptions

def generate_class_keywords(
    class_name: str
) -> List[str]:
    redis_key = f"class_keywords:{class_name}"
    keywords = get_cached_value(redis_client, redis_key)
    if type(keywords) == str:
        keywords = keywords.split(',')   

    if not keywords:
        keywords_list = generate_keywords(class_name)   
        return keywords_list

    return keywords

def initialize_document_classes() -> Dict[str, List[str]]:
    document_classes = {cls: {} for cls in document_categories}

    for cls in document_categories:
        
        redis_key = f"class_descriptions:{cls}"
        descriptions = get_cached_value(redis_client, redis_key)
        
        if not descriptions:
            descriptions = generate_class_descriptions(
                class_name=cls.replace('_', ' '),
                num_descriptions=3,
                model="gpt-4o-mini",
                temperature=0.7
            )
            cache_value(redis_client, redis_key, descriptions)
        document_classes[cls]['descriptions'] = descriptions

        redis_key = f"class_keywords:{cls}"
        keywords = get_cached_value(redis_client, redis_key)
        if type(keywords) == str:
            keywords = keywords.split(',')
        
        if not keywords:
            keywords = generate_class_keywords(
                class_name=cls.replace('_', ' '),
                model="gpt-4o-mini",
                temperature=0.7
            )
            cache_value(redis_client, redis_key, keywords)
        document_classes[cls]['keywords'] = keywords
    return document_classes


def extract_text_from_excel(file_path):
    excel_file = pd.ExcelFile(file_path)
    text_content = ''
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        text_content += df.to_string()
    return text_content


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return text

def preprocess_text(text):
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    
    return text


def get_clip_class_embeddings(document_classes, clip_model, device):
    class_embeddings = {}
    for cls, value in document_classes.items():
        embeddings = []
        descriptions = value['descriptions']
        for desc in descriptions:
            text_tokens = clip.tokenize([desc], truncate=True).to(device) 
            
            with torch.no_grad():
                text_embedding = clip_model.encode_text(text_tokens)
            
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            
            embeddings.append(text_embedding.cpu().numpy()[0])
        
        aggregated_embedding = np.mean(embeddings, axis=0)
        
        aggregated_embedding /= np.linalg.norm(aggregated_embedding)
        
        class_embeddings[cls] = aggregated_embedding
    
    return class_embeddings

def get_sbert_class_embeddings(sbert_model, document_classes):
    class_embeddings = {}
    for cls, value in document_classes.items():
        keywords = value['keywords']
        processed_descriptions = [preprocess_text(keyword) for keyword in keywords]
        embeddings = sbert_model.encode(processed_descriptions, normalize_embeddings=True)
        aggregated_embedding = np.mean(embeddings, axis=0)
        class_embeddings[cls] = aggregated_embedding
    return class_embeddings



def classify_text_document(text, class_embeddings, model, threshold=0.3):
    embedding = model.encode(text, normalize_embeddings=True)
    similarities = {cls: cosine_similarity([embedding], [emb])[0][0] for cls, emb in class_embeddings.items()}
    best_class = max(similarities, key=similarities.get)
    best_similarity = similarities[best_class]
    if best_similarity >= threshold:
        return best_class, best_similarity, similarities
    else:
        return 'unknown', best_similarity, similarities


def classify_image_document(image_path, class_embeddings, model, preprocess, device, threshold=0.3):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image)
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    image_embedding = image_embedding.cpu().numpy()[0]
    
    similarities = {cls: cosine_similarity([image_embedding], [emb])[0][0] for cls, emb in class_embeddings.items()}
    print(similarities)
    best_class = max(similarities, key=similarities.get)
    best_similarity = similarities[best_class]
    if best_similarity >= threshold:
        return best_class, best_similarity, similarities
    else:
        return 'unknown', best_similarity, similarities 

def classify_document(file_path, file_type, sbert_class_embeddings, clip_class_embeddings):
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
        return classify_text_document(text, sbert_class_embeddings, sbert_model)
    elif file_type == 'docx':
        text = extract_text_from_docx(file_path)
        return classify_text_document(text, sbert_class_embeddings, sbert_model)
    elif file_type == 'xlsx':
        text = extract_text_from_excel(file_path)
        return classify_text_document(text, sbert_class_embeddings, sbert_model)
    elif file_type == 'jpg' or file_type == 'png':
        return classify_image_document(file_path, clip_class_embeddings, clip_model, clip_preprocess, device)
    else:
        return 'unknown', 0.0
    

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

document_classes = initialize_document_classes()
clip_class_embeddings = get_clip_class_embeddings(document_classes=document_classes, clip_model=clip_model, device=device)
sbert_class_embeddings = get_sbert_class_embeddings(sbert_model, document_classes=document_classes)    