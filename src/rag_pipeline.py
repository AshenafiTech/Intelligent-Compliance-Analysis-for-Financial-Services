import pickle
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Load vector store and metadata
index = faiss.read_index("vector_store/complaints_faiss.index")
with open("vector_store/metadata.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Create sample data if not exists
if not os.path.exists("data/processed/filtered_complaints.csv"):
    os.makedirs("data/processed", exist_ok=True)
    import pandas as pd
    # Create sample complaint data
    sample_data = {
        'cleaned_narrative': [
            "I applied for a credit card but was denied without proper explanation. The process was confusing and customer service was unhelpful.",
            "My BNPL payment was processed twice causing overdraft fees. I contacted support multiple times but no resolution.",
            "Personal loan interest rates were not disclosed upfront. Hidden fees appeared after approval.",
            "Money transfer failed but funds were still deducted from my account. Took weeks to get refund.",
            "Savings account was closed without notice. Lost access to my funds for several days."
        ],
        'Product': ['Credit card', 'Buy Now, Pay Later', 'Personal loan', 'Money transfer', 'Savings account']
    }
    df = pd.DataFrame(sample_data)
    df.to_csv("data/processed/filtered_complaints.csv", index=False)
else:
    import pandas as pd
    df = pd.read_csv("data/processed/filtered_complaints.csv")

# Load embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

# Initialize OpenAI client (optional)
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        client = OpenAI(api_key=api_key)
    else:
        client = None
except Exception:
    client = None

def retrieve_chunks(question, k=5):
    """Embed question and retrieve top-k most similar chunks."""
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    retrieved = []
    for idx in I[0]:
        meta = metadatas[idx]
        chunk_text = df.loc[meta["complaint_id"], "cleaned_narrative"]
        retrieved.append({
            "chunk": chunk_text,
            "meta": meta
        })
    return retrieved

def build_prompt(question, retrieved_chunks):
    """Format the prompt for the LLM."""
    context = "\n\n".join([c["chunk"] for c in retrieved_chunks])
    prompt = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt

def generate_answer(prompt):
    """Generate answer using OpenAI GPT or fallback to simple extraction."""
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant for CrediTrust Financial. Provide concise, evidence-based answers about customer complaints."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    else:
        # Fallback: Extract key issues from context
        context = prompt.split('Context:')[1].split('Question:')[0].strip()
        question = prompt.split('Question:')[1].split('Answer:')[0].strip()
        
        # Simple keyword analysis
        issues = []
        if 'bnpl' in question.lower() or 'buy now pay later' in question.lower():
            if 'processed twice' in context.lower():
                issues.append('duplicate payments causing overdraft fees')
            if 'late fees' in context.lower():
                issues.append('unexpected late fees')
            if 'support' in context.lower() and 'no resolution' in context.lower():
                issues.append('poor customer support resolution')
        
        if issues:
            return f"Key BNPL issues identified: {', '.join(issues)}. For detailed AI analysis, configure OpenAI API key."
        else:
            return f"Analysis shows customer concerns in the retrieved complaints. Configure OpenAI API key for detailed insights."

def rag_qa(question, k=5):
    retrieved = retrieve_chunks(question, k)
    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt)
    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": retrieved[:3]  # Show top 3 sources
    }

def demo_rag():
    """Demo function to test RAG without API key"""
    questions = [
        "Why are customers unhappy with Buy Now, Pay Later?",
        "What are the main credit card issues?",
        "What problems do customers face with money transfers?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        result = rag_qa(q)
        print(f"A: {result['answer']}")
        print("Sources:")
        for i, src in enumerate(result['retrieved_sources'][:2], 1):
            print(f"  {i}. {src['chunk'][:100]}...")
        print("-" * 50)