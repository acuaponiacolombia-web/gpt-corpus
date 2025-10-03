# app.py  —  Modo CARGA-SOLO (NO REINDEXA)
# - Carga un índice ya construido en ./storage
# - Sirve PDFs de ./data en /files/<archivo.pdf>
# - Endpoint /search devuelve fragmentos + metadatos (archivo, página, score, url)
# - No usa sources.csv

from typing import List, Optional
import os
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# --- Rutas de trabajo ---
DATA_DIR = "data"
PERSIST_DIR = "storage"

# --- Modelos de E/S ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class Source(BaseModel):
    file_name: str
    page_label: Optional[str] = None
    score: Optional[float] = None
    url: Optional[str] = None  # URL servida por esta API (si el PDF está en ./data)

class SearchHit(BaseModel):
    content: str
    source: Source

class SearchResponse(BaseModel):
    results: List[SearchHit]


app = FastAPI(title="Corpus API", version="1.0.0")

# Servir archivos estáticos (PDFs) desde ./data en /files/...
if os.path.isdir(DATA_DIR):
    app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")


def _load_existing_index():
    """Carga un índice FAISS previamente persistido en ./storage.
    NO reconstruye. Si no existe, lanza error claro.
    """
    # Comprobación mínima de existencia
    must_exist = os.path.join(PERSIST_DIR, "default__vector_store.json")
    if not os.path.exists(must_exist):
        raise FileNotFoundError(
            "No se encontró un índice en ./storage. "
            "Sube los archivos del índice (default__vector_store.json, docstore.json, "
            "index_store.json, etc.) o reconstrúyelo en local y vuelve a desplegar."
        )

    # Embeddings (deben coincidir con los usados al construir el índice)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Cargar vector store + contexto desde disco
    vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=PERSIST_DIR
    )

    # Cargar índice persistido
    index = load_index_from_storage(storage_context)
    return index


@app.on_event("startup")
def _startup():
    try:
        app.state.index = _load_existing_index()
        print("[READY] Índice cargado desde ./storage.")
    except Exception as e:
        # Imprime en logs y deja fallar con mensaje claro en /health
        app.state.index = None
        print(f"[ERROR] No se pudo cargar el índice: {e}")


@app.get("/health")
def health():
    if app.state.index is None:
        return {
            "status": "error",
            "detail": "No se pudo cargar el índice en el arranque. "
                      "Verifica que ./storage contiene los archivos del índice."
        }
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request):
    if app.state.index is None:
        raise HTTPException(
            status_code=503,
            detail="Índice no disponible. Revisa /health y asegúrate de subir ./storage con el índice."
        )

    retriever = app.state.index.as_retriever(similarity_top_k=req.top_k)
    nodes = retriever.retrieve(req.query)

    results: List[SearchHit] = []
    for n in nodes:
        md = n.node.metadata or {}
        fname = md.get("file_name") or md.get("filename") or "desconocido"

        # URL servida por esta API si el PDF existe en ./data
        served_url = None
        pdf_path = os.path.join(DATA_DIR, fname)
        if os.path.exists(pdf_path):
            # /files/<archivo.pdf> (con escape por si hay espacios/caracteres)
            served_url = str(request.url_for("files", path=quote(fname)))

        hit = SearchHit(
            content=n.node.get_content(),
            source=Source(
                file_name=fname,
                page_label=str(md.get("page_label")) if md.get("page_label") else None,
                score=float(getattr(n, "score", 0.0)) if getattr(n, "score", None) is not None else None,
                url=served_url,
            ),
        )
        results.append(hit)

    return SearchResponse(results=results)
