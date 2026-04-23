import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Send query + retrieved chunks to Groq and get a cited answer."""
    
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[Chunk {i+1}]:\n{chunk['chunk']}\n\n"

    prompt = f"""You are a legal document assistant. 
Use ONLY the context below to answer the question.
Always cite which chunk your answer comes from.
If the answer is not in the context, say 'I could not find this in the document.'

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content