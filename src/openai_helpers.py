import logging
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not openai.api_key:
    logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    raise EnvironmentError("OpenAI API key not found.")


def generate_description(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 100,
    temperature: float = 0.7
) -> str:
    try:
        logging.info(f"Generating description with prompt: {prompt}")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
    return ""

def generate_keywords(
    class_name: str,
    document_text: str = "",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> str:
    prompt = (
        f"Generate a list of keywords for a '{class_name}' document that would maximize similarity "
        f"between the keywords' embeddings and the document's content embeddings. Here is an example "
        f"text/image pair from the document to help understand the context:\n\n"
        f"---\n{document_text}\n---\n\n"
        f"Please provide a concise, comma-separated list of relevant keywords."
    )

    try:
        logging.info(f"Generating keywords with prompt: {prompt}")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", 
                 "content": prompt
                 }
            ],
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
    return ""