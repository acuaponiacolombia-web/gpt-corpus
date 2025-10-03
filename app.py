# app.py
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

PERSIST_DIR = "./storage"

# --- Modelos de petición/respuesta ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class Source(BaseModel):
    file_name: str
    page_label: Optional[str] = None
    score: Optional[float] = None

class SearchHit(BaseModel):
    content: str
    source: Source

class SearchResponse(BaseModel):
    results: List[SearchHit]

# --- Inicialización (embeddings HF gratis + cargar índice FAISS) ---
app = FastAPI(title="Corpus API", version="1.0.0")

@app.on_event("startup")
def _load_index():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=PERSIST_DIR
    )
    app.state.index = load_index_from_storage(storage_context)  # solo retrieval

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    retriever = app.state.index.as_retriever(similarity_top_k=req.top_k)
    nodes = retriever.retrieve(req.query)
    results = []
    for n in nodes:
        md = n.node.metadata or {}
        results.append(SearchHit(
            content=n.node.get_content(),
            source=Source(
                file_name=md.get("file_name") or md.get("filename") or "desconocido",
                page_label=str(md.get("page_label")) if md.get("page_label") else None,
                score=float(n.score) if hasattr(n, "score") and n.score is not None else None
            )
        ))
    return SearchResponse(results=results)
