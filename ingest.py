# ingest.py
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

DATA_DIR = Path("./data")
PERSIST_DIR = "./storage"

# 1) Modelo de embeddings gratuito (HF)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# 2) Troceo de texto
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# 3) Carga PDFs con metadatos de archivo y página
# SimpleDirectoryReader añade automáticamente metadata como file_name y page_label para PDFs.
# (siempre que haya texto extraíble/O CR). 
documents = SimpleDirectoryReader(
    input_dir=str(DATA_DIR),
    recursive=True,
    required_exts=[".pdf"],
).load_data()

# 4) Crear FAISS con dimensión correcta
d = len(embed_model.get_text_embedding("hola mundo"))
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5) Construir índice y persistir a disco
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
index.storage_context.persist(persist_dir=PERSIST_DIR)
print("Índice FAISS guardado en", PERSIST_DIR)
