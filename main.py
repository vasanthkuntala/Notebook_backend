import os
import json
import requests
import asyncpg
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from PyPDF2 import PdfReader
import tiktoken

# ✅ PostgreSQL Connection
DATABASE_URL = "postgresql://notebook_lish_user:niBCJU8PwlG0yqSgOWqVyqrj3QtIk7GT@dpg-cvd8d8l2ng1s73drdm0g-a/notebook_lish"


GROQ_API_KEY = "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"

# ✅ FastAPI App
app = FastAPI()

# ✅ PostgreSQL Connection Pool
async def create_pool():
    app.state.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)

async def get_db():
    async with app.state.db_pool.acquire() as conn:
        yield conn

# ✅ Pydantic Model
class QueryRequest(BaseModel):
    query: str

# ✅ Embedding Function (Optimized)
def embed_text(text: str) -> List[float]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)[:300]  # ✅ Limit tokens for efficiency
    return [len(tokens) * 0.01] * min(10, len(tokens))  # ✅ Simulated embedding

# ✅ Cosine Similarity Function
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a**2 for a in vec1) ** 0.5
    magnitude2 = sum(b**2 for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0

# ✅ Query AI Model
@app.post("/query/")
async def query_docs(request: QueryRequest, db=Depends(get_db)):
    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long. Limit to 500 characters.")

    query_embedding = embed_text(request.query)

    # ✅ Retrieve stored document embeddings
    rows = await db.fetch("SELECT text, embedding FROM embeddings")
    
    # ✅ Compute cosine similarity
    ranked_chunks = sorted(
        rows, key=lambda row: cosine_similarity(query_embedding, json.loads(row["embedding"])),
        reverse=True
    )[:3]

    retrieved_chunks = [row["text"] for row in ranked_chunks]

    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="⚠️ No relevant data found.")

    context = "\n".join(retrieved_chunks)

    # ✅ Query Groq AI
    prompt = f"Based on these document excerpts:\n\n{context}\n\nAnswer: {request.query}"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "system", "content": "You are an AI assistant."}, {"role": "user", "content": prompt}]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        groq_answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from AI")
    else:
        groq_answer = f"Error: {response.text}"

    return {"answer": groq_answer}

# ✅ Optimized PDF Text Extraction
async def extract_text_from_pdf(file: UploadFile):
    reader = PdfReader(file.file)
    extracted_text = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
        if sum(len(t) for t in extracted_text) > 2000:
            break  # ✅ Limit memory usage

    return " ".join(extracted_text)

# ✅ Upload & Process Multiple PDFs (Clears DB Before Upload)
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...), db=Depends(get_db)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Cannot upload more than 10 files at a time.")

    # ✅ Delete all existing records before inserting new ones
    await db.execute("DELETE FROM embeddings")
    print("🗑️ Database cleared before new uploads.")

    results = []
    for file in files:
        print(f"📂 Processing: {file.filename}")

        if file.size == 0:
            results.append({"file": file.filename, "status": "Failed", "reason": "Empty file"})
            continue

        await file.seek(0)  # ✅ Reset file pointer

        extracted_text = await extract_text_from_pdf(file)

        if not extracted_text.strip():
            results.append({"file": file.filename, "status": "Failed", "reason": "No extractable text"})
            continue

        embedding = json.dumps(embed_text(extracted_text))
        await db.execute("INSERT INTO embeddings (text, embedding) VALUES ($1, $2)", extracted_text, embedding)

        results.append({"file": file.filename, "status": "Success"})

    return {"message": "Processing complete, previous records deleted", "results": results}

# ✅ Initialize Database on Startup
@app.on_event("startup")
async def startup():
    await create_pool()
