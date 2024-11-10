# Approach for Document Classification Using Embeddings and Similarity Metrics

This document classification approach leverages pre-trained embeddings and similarity metrics rather than large, labeled datasets or retraining cycles, allowing for efficient and scalable document categorization. By using pre-trained models like CLIP and Sentence Transformers, it can classify documents (e.g., bank statements, driver’s licenses) based on semantic understanding, comparing embeddings of document content with predefined class descriptions and keywords, and identifying the best match through cosine similarity. This eliminates the need for extensive data collection and resource-intensive model training while enabling flexible, accurate classification.

Adding new document types or expanding into different industries (e.g., from finance to healthcare) requires only generating new keywords and descriptions specific to the new category, bypassing the need for domain-specific training. 

This method is computationally efficient and scales well to high document volumes by performing lightweight embedding generation and similarity calculations. 

## 1. Class Description and Keyword Generation

### OpenAI GPT Models
The system uses OpenAI's GPT models to generate detailed descriptions and relevant keywords for each document category.

### CLIP Model Descriptions
Descriptions are specifically generated for the `CLIP` model because CLIP has been trained on image-description (text-image pair) data. These descriptions provide contextual information that aligns with the image-based training of CLIP, enhancing its ability to accurately classify image documents.

### Sentence Transformer Keywords
Since Sentence Transformers are text-to-text models, generating full descriptive sentences could introduce too many irrelevant words, potentially diluting the model's effectiveness. Towards addressing this, the system generates concise keywords for each category. These keywords focus on the most relevant terms, ensuring that the text-based model remains precise and efficient in classification without being overwhelmed by unnecessary information.

### Example-Based Keyword Generation
To enhance the relevance and accuracy of the generated keywords, the system provides text from an example file to the model. This context helps the model understand the specific characteristics and content of each document category, leading to the generation of more targeted and meaningful keywords.

### Caching with Redis
Generated descriptions and keywords are cached in Redis. By caching frequently used embeddings and descriptions with Redis, the system minimizes repetitive computations, further improving its capacity to handle large volumes without compromising performance.

## 2. Embedding Generation

### Sentence Transformers
For textual data, the system uses the `SentenceTransformer` model to generate embeddings from the extracted text. By utilising keywords, the embeddings are more focused and relevant.

### CLIP Model
For image data, the system utilizes OpenAI's CLIP model to generate image embeddings based on the generated descriptions tailored for image-text alignment.

### Class Embeddings
Class embeddings are created by averaging the embeddings of generated descriptions and keywords for each category, ensuring a robust representation for comparison during classification.

## 3. Document Processing

### Text Extraction
The system includes functions to extract text from PDFs, Word documents, and Excel files.

### Preprocessing
Extracted text is preprocessed to remove headers, footers, special characters, and extra whitespace, ensuring clean and consistent input for embedding generation.

### Normalization
Both text and image embeddings are normalized to ensure consistent similarity computations, facilitating accurate classification.

## 4. Classification

### Cosine Similarity
The system calculates the cosine similarity between the document embeddings and class embeddings to determine the closest matching category.

### Thresholding
A similarity threshold determines whether a document is confidently classified into a category or marked as "unknown," providing flexibility in handling ambiguous cases.

### API Endpoint
A Flask application provides an API endpoint (`/classify_file`) for uploading and classifying documents, enabling easy integration and usage.

## 5. Supported File Types
The system supports the following file extensions:

Text Documents: pdf, docx, xlsx\
Image Files: jpg, png

# Instructions to Run the Solution

## 1. Prerequisites

- **Python 3.7 or Higher**
- **Redis Server**: Ensure Redis is installed and running.
- **Environment Variables**: Set up a `.env` file or environment variables with the following keys:
  - `OPENAI_API_KEY`: Your OpenAI API key.
  - `REDIS_HOST`: Hostname for the Redis server.
  - `REDIS_PORT`: Port for the Redis server.
  - `REDIS_PASSWORD`: Password for Redis if required.

## 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Run the Flask app:
    python -m src.app

## 4. Test the classifier using a tool like curl:
    curl -X POST -F 'file=@path_to_pdf.pdf' http://127.0.0.1:5000/classify_file

## 5. Run tests:
   ```shell
    pytest
   ```
