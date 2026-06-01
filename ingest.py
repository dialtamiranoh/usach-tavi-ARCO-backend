import os
import json
import hashlib
from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuracion
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
INGESTED_LOG = BASE_DIR / "chroma_db" / "ingested_files.json"
COLLECTION_NAME = "arco_knowledge"

# Inicializar ChromaDB persistente
client = PersistentClient(path=str(CHROMA_DIR))
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# Splitter de texto
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def get_file_hash(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_ingested_log() -> dict:
    if INGESTED_LOG.exists():
        with open(INGESTED_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_ingested_log(log: dict):
    with open(INGESTED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def ingest_knowledge_json(log: dict) -> dict:
    file_hash = get_file_hash(KNOWLEDGE_PATH)
    filename = "knowledge.json"

    if log.get(filename) == file_hash:
        print(f"  {filename} sin cambios, omitiendo.")
        return log

    print("Ingestando knowledge.json...")
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    ids = []
    metadatas = []

    for item in data:
        texto = (
            f"tramite: {item['titulo']}\n"
            f"respuesta: {item['respuesta']}\n"
            f"costo: {item.get('costo', 'no especificado')}\n"
            f"duracion: {item.get('duracion', 'no especificada')}\n"
            f"canal: {item['canal']}\n"
            f"presencialidad: {item['presencialidad']}\n"
            f"requiere clave unica: {item['requiere_clave_unica']}\n"
            f"fuente: {item['fuente']}\n"
            f"keywords: {', '.join(item['keywords'])}"
        )

        chunks.append(texto)
        ids.append(item['id'])
        metadatas.append({
                    "source": "knowledge.json",
                    "tramite_id": item['id'],
                    "titulo": item['titulo'],
                    "fuente": item['fuente'],
                    "costo": item.get('costo', ''),
                    "duracion": item.get('duracion', ''),
                    "canal": item['canal'],
                    "presencialidad": item['presencialidad'],
                    "requiere_clave_unica": item['requiere_clave_unica']
        })

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"  {len(chunks)} tramites ingestados desde knowledge.json")
    log[filename] = file_hash
    return log

def ingest_file(filepath: Path, log: dict) -> dict:
    file_hash = get_file_hash(filepath)
    filename = filepath.name

    if log.get(filename) == file_hash:
        print(f"  {filename} sin cambios, omitiendo.")
        return log

    print(f"Ingestando: {filename}")
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    chunks_text = splitter.split_text(text)
    ids = [f"{filepath.stem}_{i}" for i in range(len(chunks_text))]
    metadatas = [{"source": filename} for _ in chunks_text]
    collection.upsert(documents=chunks_text, ids=ids, metadatas=metadatas)
    print(f"  {len(chunks_text)} chunks agregados desde {filename}")
    log[filename] = file_hash
    return log

def ingest_all():
    log = load_ingested_log()

    # Ingestar knowledge.json
    log = ingest_knowledge_json(log)

    # Ingestar documentos adicionales desde docs/
    if DOCS_DIR.exists():
        files = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))
        if files:
            for f in files:
                log = ingest_file(f, log)
        else:
            print("No hay documentos adicionales en docs/")

    save_ingested_log(log)
    print(f"\nTotal documentos en coleccion: {collection.count()}")

if __name__ == "__main__":
    ingest_all()