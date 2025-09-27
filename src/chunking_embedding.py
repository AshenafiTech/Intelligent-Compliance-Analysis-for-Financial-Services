import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load cleaned data
if not os.path.exists('data/processed/filtered_complaints.csv'):
    os.makedirs('data/processed', exist_ok=True)
    # Create sample complaint data for demo
    sample_data = {
        'cleaned_narrative': [
            "I applied for a credit card but was denied without proper explanation. The process was confusing and customer service was unhelpful.",
            "My BNPL payment was processed twice causing overdraft fees. I contacted support multiple times but no resolution.",
            "Personal loan interest rates were not disclosed upfront. Hidden fees appeared after approval.",
            "Money transfer failed but funds were still deducted from my account. Took weeks to get refund.",
            "Savings account was closed without notice. Lost access to my funds for several days.",
            "Credit card application process took too long and communication was poor throughout.",
            "BNPL service charged unexpected late fees even though payment was made on time.",
            "Personal loan approval was delayed causing me to miss important financial deadlines."
        ],
        'Product': ['Credit card', 'Buy Now, Pay Later', 'Personal loan', 'Money transfer', 'Savings account', 'Credit card', 'Buy Now, Pay Later', 'Personal loan']
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('data/processed/filtered_complaints.csv', index=False)
else:
    df = pd.read_csv('data/processed/filtered_complaints.csv')

# Chunking strategy
chunk_size = 300  # Experimented and found a balance between context and granularity
chunk_overlap = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Prepare chunks and metadata
chunks = []
metadatas = []
for idx, row in df.iterrows():
    text = str(row['cleaned_narrative'])
    splits = splitter.split_text(text)
    for i, chunk in enumerate(splits):
        chunks.append(chunk)
        metadatas.append({
            "complaint_id": idx,
            "product": row['Product'],
            "chunk_index": i
        })

# Embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/complaints_faiss.index")
with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print(f"Indexed {len(chunks)} chunks. Vector store saved in vector_store/")