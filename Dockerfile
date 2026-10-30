FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# Copy project files
COPY app.py .
COPY configs/ configs/
COPY generation/ generation/
COPY vector_store/ vector_store/
COPY preprocessing/ preprocessing/

# Pre-download embedding model during build — eliminates cold start delay
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')"

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
