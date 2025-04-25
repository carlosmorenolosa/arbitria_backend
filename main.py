import os
import re
from pathlib import Path
from typing import List, Literal

import google.generativeai as genai
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# StaticFiles removed: now PDFs served from S3
from pydantic import BaseModel
from dotenv import load_dotenv


# Carga las variables definidas en backend/.env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── Configuración ───────────────────────────────────────────────────────────
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "reglamento-arbitral")
# Leemos la URL base del bucket S3
PDF_BASE_URL = os.getenv("PDF_BASE_URL")

if not (GENAI_API_KEY and PINECONE_API_KEY):
    raise RuntimeError("Faltan GENAI_API_KEY o PINECONE_API_KEY en el entorno.")
if not PDF_BASE_URL:
    raise RuntimeError("Falta PDF_BASE_URL en el entorno.")

# Inicializa APIs
genai.configure(api_key=GENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc = None
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = genai.GenerativeModel("gemini-2.0-flash")

# ─── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Asistente de Reglamento Arbitral")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # permitir cualquier origen
    allow_methods=["*"],        # permitir todos los métodos (GET, POST, OPTIONS…)
    allow_headers=["*"],        # permitir todos los headers
)
# Ya no montamos carpeta local StaticFiles

# ─── Modelos Pydantic ─────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

class FragmentOut(BaseModel):
    nombre_original: str
    texto: str
    page: int | None = None
    pdf_url: str

class ChatResponse(BaseModel):
    answer: str
    fragments: List[FragmentOut]

# ─── Funciones RAG ────────────────────────────────────────────────────────────
def vectorize_query(text: str) -> List[float]:
    emb = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return emb["embedding"]


def retrieve_fragments(query_vector, top_k=5):
    res = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return res.matches


def format_prompt(query: str, fragments, history: List[ChatMessage]) -> str:
    context = "\n\n".join(
        f"Fragmento {i+1} (de {m.metadata.get('nombre_original','documento desconocido')}):\n{m.metadata.get('texto','')}"
        for i, m in enumerate(fragments)
    )
    hist = "\n\n".join(
        f"{'Usuario' if h.role=='user' else 'Asistente'}: {h.content}" for h in history
    )
    prompt = f"""
Eres un asistente especializado en reglamentos arbitrales. Tu objetivo es proporcionar respuestas precisas y útiles basadas únicamente en los documentos de referencia proporcionados.

INSTRUCCIONES:
1. Analiza cuidadosamente la consulta del usuario.
2. Utiliza ÚNICAMENTE la información proporcionada en los fragmentos de contexto para responder.
3. Si la información en los fragmentos no es suficiente para responder la consulta completa, indica qué partes puedes responder y qué información no dispones.
4. No inventes ni supongas información que no esté en los fragmentos.
5. Cita las fuentes de manera clara.
6. Responde en español con un tono profesional pero accesible.
7. Si la pregunta no está relacionada con reglamentos arbitrales, indica amablemente que estás especializado en ese tema.

CONTEXTO DE FRAGMENTOS RECUPERADOS:
{context}

HISTORIAL DE LA CONVERSACIÓN:
{hist}

CONSULTA DEL USUARIO:
{query}

RESPUESTA:
"""
    return prompt

# ─── Endpoint principal ───────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        q_vec = vectorize_query(req.query)
        frags = retrieve_fragments(q_vec, top_k=5)
    except Exception as e:
        raise HTTPException(500, f"Error al recuperar fragments: {e}")

    full_prompt = format_prompt(req.query, frags, req.history)
    try:
        gen = model.generate_content(full_prompt)
        answer = gen.candidates[0].content.parts[0].text
    except Exception as e:
        raise HTTPException(500, f"Error generando respuesta: {e}")

    parsed: List[FragmentOut] = []
    for m in frags:
        fname = m.metadata.get("nombre_original", "unk.pdf")
        # Construye URL pública apuntando a S3
        pdf_url = f"{PDF_BASE_URL}/{fname}"
        page = m.metadata.get("page")
        parsed.append(FragmentOut(
            nombre_original=fname,
            texto=m.metadata.get("texto", ""),
            page=page,
            pdf_url=pdf_url
        ))

    return ChatResponse(answer=answer, fragments=parsed)

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
