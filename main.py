"""
main.py — Backend unificado ARCO
Puerto: 8000  →  uvicorn main:app --port 8000 --reload

Modelos LLM disponibles:
  granite → http://127.0.0.1:8001  (Granite-4.0-1B)
  qwen    → http://127.0.0.1:8002  (Qwen2.5-1.5B)

Archivos de persistencia:
  benchmark_metrics.jsonl  — una línea por consulta (campo model_key)
  benchmark_feedback.jsonl — feedback por trace_id
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Literal
import json
import unicodedata
import requests
import re
import time
import uuid
from datetime import datetime
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("arco.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("arco")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

load_dotenv()

app = FastAPI(title="ARCO API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rutas y archivos
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
METRICS_PATH   = BASE_DIR / "benchmark_metrics.jsonl"
FEEDBACK_PATH  = BASE_DIR / "benchmark_feedback.jsonl"

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------
MODELS = {
    "granite": {
        "url":   os.getenv("GRANITE_URL",   "http://127.0.0.1:8001/v1/chat/completions"),
        "model": os.getenv("GRANITE_MODEL", "granite-4.0-1b-instruct"),
        "label": "Granite-4.0-1B",
    },
    "qwen": {
        "url":   os.getenv("QWEN_URL",   "http://127.0.0.1:8002/v1/chat/completions"),
        "model": os.getenv("QWEN_MODEL", "qwen2.5-1.5b-instruct"),
        "label": "Qwen2.5-1.5B",
    },
}

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",  "140"))

# Cargar prompts al iniciar
_prompts_path = BASE_DIR / "prompts.json"
with open(_prompts_path, "r", encoding="utf-8") as _f:
    _prompts_data = json.load(_f)
_version_activa = os.getenv("PROMPT_VERSION", _prompts_data["version_activa"])
SYSTEM_PROMPT = _prompts_data["versiones"][_version_activa]["system_prompt"]

# ---------------------------------------------------------------------------
# RAG (opcional — activa con USE_RAG=true en .env)
# ---------------------------------------------------------------------------
USE_RAG = os.getenv("USE_RAG", "false").lower() == "true"

if USE_RAG:
    from chromadb import PersistentClient
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    CHROMA_DIR      = BASE_DIR / "chroma_db"
    COLLECTION_NAME = "arco_knowledge"

    chroma_client  = PersistentClient(path=str(CHROMA_DIR))
    embedding_fn   = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    chroma_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    def search_rag(query: str, n_results: int = 3) -> list[dict]:
        results = chroma_collection.query(query_texts=[query], n_results=n_results)
        if not results["documents"][0]:
            return []
        items = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            items.append({
                "texto":                doc,
                "source":               meta.get("source", ""),
                "titulo":               meta.get("titulo", ""),
                "fuente":               meta.get("fuente", ""),
                "costo":                meta.get("costo"),
                "duracion":             meta.get("duracion"),
                "canal":                meta.get("canal"),
                "presencialidad":       meta.get("presencialidad"),
                "requiere_clave_unica": meta.get("requiere_clave_unica"),
            })
        return items

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ContextData(BaseModel):
    tramite:              Optional[str] = None
    respuesta_base:       Optional[str] = None
    costo:                Optional[str] = None
    duracion:             Optional[str] = None
    canal:                Optional[str] = None
    presencialidad:       Optional[str] = None
    requiere_clave_unica: Optional[str] = None
    fuente:               Optional[str] = None


class Question(BaseModel):
    query:    str
    model:    Literal["granite", "qwen"] = "granite"
    history:  list[ChatMessage] = Field(default_factory=list)
    context:  Optional[ContextData] = None
    trace_id: Optional[str] = None


class Feedback(BaseModel):
    trace_id: str
    model:    str
    score:    float    # 1.0 positivo · 0.5 neutral · 0.0 negativo
    comment:  Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def clean_model_text(text: str) -> str:
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace('"', "").replace("•", " ").replace("*", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_followup_query(query: str) -> bool:
    q = normalize_text(query)
    hints = [
        "y eso", "eso", "ese tramite", "esa gestion", "ese documento",
        "se puede", "requiere", "necesita", "necesito", "clave unica",
        "presencial", "online", "en linea", "por internet",
        "cuanto", "cuesta", "costo", "donde", "como", "cuando",
        "que necesito", "que documentos", "y si", "y para eso",
    ]
    return any(h in q for h in hints)


def context_to_item(ctx: ContextData) -> Optional[dict]:
    if not ctx or not ctx.tramite or not ctx.respuesta_base:
        return None
    return {
        "titulo":               ctx.tramite,
        "respuesta":            ctx.respuesta_base,
        "costo":                ctx.costo,
        "duracion":             ctx.duracion,
        "canal":                ctx.canal,
        "presencialidad":       ctx.presencialidad,
        "requiere_clave_unica": ctx.requiere_clave_unica,
        "fuente":               ctx.fuente,
    }


def build_history_text(history: list[ChatMessage]) -> str:
    if not history:
        return "sin historial previo"
    return "\n".join(f"{m.role}: {m.content}" for m in history[-6:])


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def save_metric(entry: dict) -> None:
    with open(METRICS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_feedback_entry(entry: dict) -> None:
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_metrics() -> list[dict]:
    if not METRICS_PATH.exists():
        return []
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_feedbacks() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Llamada al LLM
# ---------------------------------------------------------------------------

def call_llm(
    user_query: str,
    item: dict,
    history: list[ChatMessage],
    model_key: str,
    using_previous_context: bool = False,
) -> tuple[str, dict]:
    cfg = MODELS[model_key]

    # Cargar prompt desde prompts.json
    system_prompt = SYSTEM_PROMPT

    history_text = build_history_text(history)
    context_text = f"""
        tramite: {item['titulo']}
        respuesta base: {item['respuesta']}
        costo: {item.get('costo', 'no especificado')}
        duracion: {item.get('duracion', 'no especificada')}
        canal: {item['canal']}
        presencialidad: {item['presencialidad']}
        requiere clave unica: {item['requiere_clave_unica']}
        fuente oficial: {item['fuente']}
    """.strip()

    followup_note = (
        "esta pregunta parece ser continuacion del mismo tramite detectado anteriormente."
        if using_previous_context
        else "esta pregunta corresponde a un tramite detectado directamente."
    )

    user_prompt = f"""
pregunta actual del usuario: {user_query}

historial reciente:
{history_text}

contexto del tramite:
{context_text}

nota:
{followup_note}

redacta una respuesta breve de orientacion para el usuario.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    input_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
    payload = {
        "model":       cfg["model"],
        "messages":    messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens":  LLM_MAX_TOKENS,
    }

    start    = time.time()
    fallback = False
    error    = None

    try:
        resp = requests.post(cfg["url"], json=payload, timeout=120)
        resp.raise_for_status()
        data     = resp.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        respuesta = clean_model_text(raw_text)

        usage = data.get("usage", {})
        if usage:
            input_tokens  = usage.get("prompt_tokens",     input_tokens)
            output_tokens = usage.get("completion_tokens", 0)
        else:
            output_tokens = estimate_tokens(respuesta)

    except Exception as e:
        respuesta     = clean_model_text(item["respuesta"])
        input_tokens  = 0
        output_tokens = 0
        fallback      = True
        error         = str(e)

    elapsed_ms = (time.time() - start) * 1000

    return respuesta, {
        "latency_ms":    elapsed_ms,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "fallback":      fallback,
        "error":         error,
        "model_key":     model_key,
        "model_label":   cfg["label"],
        "model_id":      cfg["model"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "ARCO backend unificado",
        "modelos": {k: v["label"] for k, v in MODELS.items()},
    }


@app.get("/models")
def get_models():
    return {
        k: {"label": v["label"], "url": v["url"], "model_id": v["model"]}
        for k, v in MODELS.items()
    }


@app.post("/ask")
def ask_question(question: Question):
    trace_id  = question.trace_id or str(uuid.uuid4())
    query     = normalize_text(question.query)
    model_key = question.model

    matched_item           = None
    using_previous_context = False

    # RAG (si está activo)
    if USE_RAG:
        rag_results = search_rag(question.query)
        if rag_results:
            mejor = rag_results[0]
            matched_item = {
                
                "titulo":               mejor["titulo"] or "Resultado RAG",
                "respuesta":            mejor["texto"],
                "costo":                mejor.get("costo"),
                "duracion":             mejor.get("duracion"),
                "canal":                mejor.get("canal", "ver fuente oficial"),
                "presencialidad":       mejor.get("presencialidad", "ver fuente oficial"),
                "requiere_clave_unica": mejor.get("requiere_clave_unica", "ver fuente oficial"),
                "fuente":               mejor["fuente"] or "",
            }
            

    # Búsqueda por keywords en knowledge.json
    if not matched_item:
        for item in KNOWLEDGE:
            for kw in item["keywords"]:
                if normalize_text(kw) in query:
                    matched_item = item
                    break
            if matched_item:
                break

    # Caso especial: licencia de conducir
    if "licencia de conducir" in query or "renovar licencia" in query:
        respuesta = (
            "La renovacion de licencia de conducir no corresponde al Registro Civil. "
            "ARCO esta enfocado en tramites del Registro Civil y su orientacion."
        )
        save_metric({
            "trace_id": trace_id, "timestamp": datetime.now().isoformat(),
            "tramite": "Fuera del alcance", "user_query": question.query,
            "respuesta": respuesta, "latency_ms": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "fallback": False, "error": None,
            "model_key": model_key, "model_label": MODELS[model_key]["label"],
            "model_id": MODELS[model_key]["model"], "using_previous_context": False,
        })
        return {
            "tramite": "Fuera del alcance de ARCO", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": "https://www.chileatiende.gob.cl/fichas/20592-licencias-de-conducir",
            "trace_id": trace_id, "model_used": MODELS[model_key]["label"],
        }

    # Seguimiento de contexto previo
    if not matched_item and question.context and is_followup_query(question.query):
        ctx_item = context_to_item(question.context)
        if ctx_item:
            matched_item           = ctx_item
            using_previous_context = True

    logger.info(
        f"query='{question.query}' "
        f"modelo={model_key} "
        f"tramite='{matched_item['titulo'] if matched_item else 'No identificado'}' "
        f"rag={USE_RAG} "
        f"followup={using_previous_context}"
    )

    # Trámite no identificado
    if not matched_item:
        respuesta = (
            "ARCO todavia no tiene informacion suficiente para orientar ese tramite "
            "dentro del alcance actual del demo."
        )
        save_metric({
            "trace_id": trace_id, "timestamp": datetime.now().isoformat(),
            "tramite": "No identificado", "user_query": question.query,
            "respuesta": respuesta, "latency_ms": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "fallback": False, "error": None,
            "model_key": model_key, "model_label": MODELS[model_key]["label"],
            "model_id": MODELS[model_key]["model"], "using_previous_context": False,
        })
        return {
            "tramite": "No identificado", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": None, "trace_id": trace_id,
            "model_used": MODELS[model_key]["label"],
        }

    # Llamada al LLM elegido
    try:
        respuesta_ia, meta = call_llm(
            question.query, matched_item, question.history,
            model_key, using_previous_context,
        )
    except Exception as e:
        respuesta_ia = clean_model_text(matched_item["respuesta"])
        meta = {
            "latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "fallback": True, "error": str(e),
            "model_key": model_key, "model_label": MODELS[model_key]["label"],
            "model_id": MODELS[model_key]["model"],
        }

    save_metric({
        "trace_id":               trace_id,
        "timestamp":              datetime.now().isoformat(),
        "tramite":                matched_item["titulo"],
        "user_query":             question.query,
        "respuesta":              respuesta_ia,
        "latency_ms":             meta["latency_ms"],
        "input_tokens":           meta["input_tokens"],
        "output_tokens":          meta["output_tokens"],
        "total_tokens":           meta["total_tokens"],
        "fallback":               meta["fallback"],
        "error":                  meta["error"],
        "model_key":              meta["model_key"],
        "model_label":            meta["model_label"],
        "model_id":               meta["model_id"],
        "using_previous_context": using_previous_context,
    })

    return {
        "tramite":               matched_item["titulo"],
        "respuesta":             respuesta_ia,
        "respuesta_base":        matched_item["respuesta"],
        "costo":                 matched_item.get("costo"),
        "duracion":              matched_item.get("duracion"),
        "canal":                 matched_item["canal"],
        "presencialidad":        matched_item["presencialidad"],
        "requiere_clave_unica":  matched_item["requiere_clave_unica"],
        "fuente":                matched_item["fuente"],
        "trace_id":              trace_id,
        "model_used":            MODELS[model_key]["label"],
        "latency_ms":            round(meta["latency_ms"], 0),
        "total_tokens":          meta["total_tokens"],
        "fallback":              meta["fallback"],
    }


@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    save_feedback_entry({
        "trace_id":  feedback.trace_id,
        "model":     feedback.model,
        "score":     feedback.score,
        "comment":   feedback.comment,
        "timestamp": datetime.now().isoformat(),
    })
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Endpoints de métricas y benchmark
# ---------------------------------------------------------------------------

@app.get("/metrics")
def get_metrics():
    metrics   = load_metrics()
    feedbacks = load_feedbacks()
    fb_index  = {fb["trace_id"]: fb for fb in feedbacks}
    for m in metrics:
        m["feedback"] = fb_index.get(m["trace_id"])
    return metrics


@app.get("/stats")
def get_stats():
    metrics = load_metrics()
    if not metrics:
        return {"error": "No hay métricas aún"}

    feedbacks = load_feedbacks()
    positive  = sum(1 for fb in feedbacks if fb["score"] >= 0.8)
    negative  = sum(1 for fb in feedbacks if fb["score"] <= 0.2)

    latencies    = [m["latency_ms"] for m in metrics if m.get("latency_ms")]
    total_tokens = sum(m.get("total_tokens", 0) for m in metrics)
    fallbacks    = sum(1 for m in metrics if m.get("fallback"))

    return {
        "total_consultas":         len(metrics),
        "promedio_latencia_ms":    round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "total_tokens_consumidos": total_tokens,
        "tasa_fallback":           round(fallbacks / len(metrics) * 100, 2),
        "feedback_positivos":      positive,
        "feedback_negativos":      negative,
        "modelos_activos":         list(MODELS.keys()),
    }


@app.get("/stats/compare")
def get_stats_compare():
    metrics   = load_metrics()
    feedbacks = load_feedbacks()
    fb_index  = {fb["trace_id"]: fb for fb in feedbacks}

    result = {}
    for key, cfg in MODELS.items():
        subset = [m for m in metrics if m.get("model_key") == key]
        if not subset:
            result[key] = {
                "label":                 cfg["label"],
                "total_consultas":       0,
                "promedio_latencia_ms":  0,
                "p95_latencia_ms":       0,
                "total_tokens":          0,
                "tokens_por_consulta":   0,
                "tasa_fallback":         0,
                "feedback_positivos":    0,
                "feedback_negativos":    0,
                "feedback_neutral":      0,
                "pct_feedback_positivo": 0,
            }
            continue

        latencies = sorted(m["latency_ms"] for m in subset if m.get("latency_ms"))
        p95_idx   = int(len(latencies) * 0.95) - 1 if latencies else 0
        p95       = latencies[max(p95_idx, 0)] if latencies else 0

        total_tok = sum(m.get("total_tokens", 0) for m in subset)
        fallbacks = sum(1 for m in subset if m.get("fallback"))

        fb_subset = [fb_index[m["trace_id"]] for m in subset if m["trace_id"] in fb_index]
        positive  = sum(1 for fb in fb_subset if fb["score"] >= 0.8)
        negative  = sum(1 for fb in fb_subset if fb["score"] <= 0.2)
        neutral   = len(fb_subset) - positive - negative
        pct_pos   = round(positive / len(fb_subset) * 100, 1) if fb_subset else 0

        result[key] = {
            "label":                 cfg["label"],
            "total_consultas":       len(subset),
            "promedio_latencia_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p95_latencia_ms":       round(p95, 1),
            "total_tokens":          total_tok,
            "tokens_por_consulta":   round(total_tok / len(subset), 1),
            "tasa_fallback":         round(fallbacks / len(subset) * 100, 2),
            "feedback_positivos":    positive,
            "feedback_negativos":    negative,
            "feedback_neutral":      neutral,
            "pct_feedback_positivo": pct_pos,
        }

    return result


@app.get("/metrics/timeline")
def get_timeline():
    metrics = load_metrics()
    result  = {}
    for key in MODELS:
        subset = [m for m in metrics if m.get("model_key") == key]
        subset.sort(key=lambda x: x.get("timestamp", ""))
        result[key] = [
            {
                "timestamp":  m["timestamp"],
                "latency_ms": round(m.get("latency_ms", 0), 1),
                "tokens":     m.get("total_tokens", 0),
                "fallback":   m.get("fallback", False),
                "tramite":    m.get("tramite", ""),
            }
            for m in subset[-50:]
        ]
    return result
