"""
main.py — Backend de producción ARCO
Puerto: 8000  →  uvicorn main:app --port 8000 --reload

Sirve un único modelo LLM (configurado via LLM_URL / LLM_MODEL_ID / LLM_LABEL
en .env). La comparación entre múltiples modelos .gguf (búsqueda, arranque,
TTFT, tokens/seg, etc.) vive en dashboard.py, un servicio independiente.
"""

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import json
import unicodedata
import requests
import re
import time
import uuid
from datetime import datetime
import os
import logging
import psutil

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
EVAL_REPORT_PATH = BASE_DIR / "eval" / "reporte_latest.json"
OFF_TOPIC_REFUSAL = "no es un tramite relacionado con este registro civil"

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

# --- Mapeo de WhatsApp Team ---
WHATSAPP_TEAM: dict[str, str] = {}
for _pair in os.getenv("WHATSAPP_TEAM", "").split(","):
    _pair = _pair.strip()
    if not _pair or ":" not in _pair:
        continue
    _name, _phone = _pair.split(":", 1)
    WHATSAPP_TEAM[_phone.strip()] = _name.strip()

def resolve_integrante(phone_number: str) -> str:
    return WHATSAPP_TEAM.get(phone_number, phone_number)

# --- Persistencia de Métricas ---
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

def base_metric_entry(
    trace_id: str, tramite: str, user_query: str, respuesta: str,
    channel: str, integrante: Optional[str], telefono: Optional[str] = None, **extra,
) -> dict:
    return {
        "trace_id":               trace_id,
        "timestamp":              datetime.now().isoformat(),
        "tramite":                tramite,
        "user_query":             user_query,
        "respuesta":              respuesta,
        "channel":                channel,
        "integrante":             integrante,
        "telefono":               telefono,
        "model_label":            MODEL["label"],
        "model_id":               MODEL["model"],
        "latency_ms":             extra.get("latency_ms", 0),
        "ttft_ms":                extra.get("ttft_ms"),
        "input_tokens":           extra.get("input_tokens", 0),
        "output_tokens":          extra.get("output_tokens", 0),
        "total_tokens":           extra.get("total_tokens", 0),
        "fallback":               extra.get("fallback", False),
        "error":                  extra.get("error"),
        "using_previous_context": extra.get("using_previous_context", False),
        "memoria_mb":             round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1),
    }


# ---------------------------------------------------------------------------
# Configuración del modelo (uno solo)
# ---------------------------------------------------------------------------
MODEL = {
    "url":   os.getenv("LLM_URL",      "http://127.0.0.1:8001/v1/chat/completions"),
    "model": os.getenv("LLM_MODEL_ID", "granite-4.0-1b-instruct"),
    "label": os.getenv("LLM_LABEL",    "Granite-4.0-1B"),
}

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",  "140"))

# ---------------------------------------------------------------------------
# Configuración WhatsApp Cloud API
# ---------------------------------------------------------------------------
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "arco_tavi_verify_token")
WHATSAPP_GRAPH_VERSION = os.getenv("WHATSAPP_GRAPH_VERSION", "v22.0")


# Cargar prompts al iniciar
_prompts_path = BASE_DIR / "prompts.json"
with open(_prompts_path, "r", encoding="utf-8") as _f:
    _prompts_data = json.load(_f)
_version_activa = os.getenv("PROMPT_VERSION", _prompts_data["version_activa"])
SYSTEM_PROMPT = _prompts_data["versiones"][_version_activa]["system_prompt"]

# ---------------------------------------------------------------------------
# RAG (opcional — activa con USE_RAG=true en .env)
# ---------------------------------------------------------------------------
USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"

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
        
        distances = results.get("distances", [[]])[0]
        items = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            # Si hay distancias registradas, descartar coincidencias con distancia > 0.8 (fuera de tema)
            if distances and i < len(distances):
                dist = distances[i]
                logger.info(f"RAG: Candidato '{meta.get('titulo')}' tiene distancia {dist:.4f}")
                if dist > 0.8:
                    continue
            
            items.append({
                "id":                   meta.get("tramite_id"),
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
    history:  list[ChatMessage] = Field(default_factory=list)
    context:  Optional[ContextData] = None
    trace_id: Optional[str] = None


class Feedback(BaseModel):
    trace_id: str
    score:    float
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


# --- Funciones NLP y Multi-Trámite (pruebas-pablo) ---
LLM_URL = MODEL["url"]
LLM_MODEL = MODEL["model"]

def build_dependency_index(knowledge: list[dict]):
    id_to_item = {}
    manual_deps = {}

    for item in knowledge:
        item_id = item.get("id")
        if not item_id:
            continue
        id_to_item[item_id] = item
        manual_deps[item_id] = set(item.get("depende_de", []))

    def transitive_manual(start_id: str) -> set[str]:
        seen = set()
        pending = list(manual_deps.get(start_id, set()))
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(manual_deps.get(current, set()))
        return seen

    clave_unica_prereqs = transitive_manual("clave_unica")

    dependencies = {}
    for item_id, item in id_to_item.items():
        deps = set(manual_deps.get(item_id, set()))
        if (
            item_id != "clave_unica"
            and item_id not in clave_unica_prereqs
            and item.get("requiere_clave_unica") in ("si", "depende")
        ):
            deps.add("clave_unica")
        dependencies[item_id] = deps

    return id_to_item, dependencies


ID_TO_ITEM, DEPENDENCIES = build_dependency_index(KNOWLEDGE)


def expand_dependencies(matched_ids: list[str]) -> list[str]:
    expanded = list(matched_ids)
    seen = set(matched_ids)
    pending = list(matched_ids)

    while pending:
        current = pending.pop(0)
        for dep in sorted(DEPENDENCIES.get(current, set())):
            if dep not in seen and dep in ID_TO_ITEM:
                seen.add(dep)
                expanded.append(dep)
                pending.append(dep)

    return expanded


def topological_order(ids: list[str]) -> list[str]:
    id_set = set(ids)
    graph = {i: {d for d in DEPENDENCIES.get(i, set()) if d in id_set} for i in ids}
    order = []
    visited = set()
    temp = set()

    def visit(node):
        if node in visited or node in temp:
            return
        temp.add(node)
        for dep in graph.get(node, set()):
            visit(dep)
        temp.discard(node)
        visited.add(node)
        order.append(node)

    for node in ids:
        visit(node)

    return order


STOP_WORDS = {
    "quiero", "como", "puedo", "sobre", "para", "este", "esta", "estos", "estas",
    "todo", "toda", "todos", "todas", "donde", "cuando", "quien", "cual", "cuales",
    "consultar", "saber", "conocer", "informacion", "ayuda", "asistente", "arco",
    "hola", "buenos", "dias", "tardes", "noches", "por", "favor", "gracias",
    "hacer", "realizar", "solicitar", "obtener", "sacar", "pedir", "ver", "buscar",
    "de", "la", "el", "en", "un", "una", "y", "o", "a", "con", "del", "al", "los", "las", "unos", "unas"
}

def has_keyword_overlap(query: str, item: dict) -> bool:
    q_words = {w for w in normalize_text(query).split() if w not in STOP_WORDS and len(w) > 2}
    if not q_words:
        return False
    
    item_words = set()
    for kw in item.get("keywords", []):
        for w in normalize_text(kw).split():
            if w not in STOP_WORDS and len(w) > 2:
                item_words.add(w)
                
    for w in normalize_text(item.get("titulo", "")).split():
        if w not in STOP_WORDS and len(w) > 2:
            item_words.add(w)
            
    return len(q_words.intersection(item_words)) > 0


def keyword_match(query_normalizada: str) -> list[str]:
    ids = []
    seen = set()
    for item in KNOWLEDGE:
        item_id = item.get("id")
        if item_id in seen:
            continue
        for keyword in item["keywords"]:
            if normalize_text(keyword) in query_normalizada:
                if item_id:
                    ids.append(item_id)
                    seen.add(item_id)
                break
    return ids


def classify_tramites_with_llm(user_query: str) -> Optional[list[str]]:
    catalogo = "\n".join(f"- {item['id']}: {item['titulo']}" for item in KNOWLEDGE)

    system_prompt = (
        "eres un clasificador de intencion para tramites del registro civil de chile. "
        "recibes una consulta de un usuario y un catalogo de tramites, cada uno con su id. "
        "tu unica tarea es identificar cuales ids del catalogo corresponden a lo que el usuario "
        "esta pidiendo o mencionando, incluyendo sinonimos y formas coloquiales de preguntar. "
        "responde EXCLUSIVAMENTE con un arreglo json de strings con los ids que correspondan, "
        "sin texto adicional, sin explicaciones, sin markdown. "
        "si ningun tramite del catalogo corresponde, responde con un arreglo vacio []. "
        "nunca inventes un id que no este en el catalogo."
    )

    user_prompt = f"""
catalogo de tramites:
{catalogo}

consulta del usuario: {user_query}

responde solo con el arreglo json de ids.
""".strip()

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 200
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
        raw_text = re.sub(r"```$", "", raw_text).strip()

        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            return None

        valid_ids = []
        seen = set()
        for candidate in parsed:
            if isinstance(candidate, str) and candidate in ID_TO_ITEM and candidate not in seen:
                valid_ids.append(candidate)
                seen.add(candidate)

        return valid_ids
    except Exception:
        return None


CONCERN_IDS = {"cedula_identidad_chilenos", "cedula_identidad_extranjeros", "clave_unica"}


def transitive_dependencies(item_id: str) -> set[str]:
    seen = set()
    pending = list(DEPENDENCIES.get(item_id, set()))
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(DEPENDENCIES.get(current, set()))
    return seen


def has_real_relation(id_a: str, id_b: str) -> bool:
    return id_b in transitive_dependencies(id_a) or id_a in transitive_dependencies(id_b)


def find_unmatched_clauses(user_query: str) -> list[str]:
    clauses = re.split(r"(?:,|\.|;| pero | y | tambien | ademas )", user_query, flags=re.IGNORECASE)

    all_keywords_norm = []
    for item in KNOWLEDGE:
        all_keywords_norm.extend(normalize_text(k) for k in item["keywords"])

    unmatched = []
    for clause in clauses:
        clause_stripped = clause.strip()
        if not clause_stripped:
            continue
        clause_norm = normalize_text(clause_stripped)
        if not any(kw in clause_norm for kw in all_keywords_norm):
            unmatched.append(clause_stripped)

    return unmatched


def extract_relevant_clause(user_query: str, item: dict) -> str:
    keywords = item.get("keywords")
    if not keywords:
        return user_query

    clauses = re.split(r"(?:,|\.|;| pero | y | tambien | ademas )", user_query, flags=re.IGNORECASE)
    keywords_norm = [normalize_text(k) for k in keywords]

    relevant = []
    for clause in clauses:
        clause_norm = normalize_text(clause)
        if any(kw in clause_norm for kw in keywords_norm):
            relevant.append(clause.strip())

    if relevant:
        return " ".join(relevant)
    return user_query


def response_has_foreign_url(texto: str, fuente_valida: Optional[str]) -> bool:
    urls = re.findall(r"https?://\S+", texto)
    fuente_normalizada = (fuente_valida or "").rstrip(".,;:")
    for url in urls:
        url_normalizada = url.rstrip(".,;:")
        if url_normalizada != fuente_normalizada:
            return True
    return False


def build_single_item_fallback(item: dict) -> str:
    partes = [clean_model_text(item["respuesta"])]
    if item.get("costo"):
        partes.append(f"Costo: {item['costo']}.")
    partes.append(f"Fuente: {item['fuente']}.")
    return " ".join(partes)


def generate_llm_response(
    user_query: str,
    item: dict,
    history: list[ChatMessage],
    using_previous_context: bool = False,
    ignorar_otros_tramites: bool = False
) -> tuple[str, dict]:
    system_prompt = SYSTEM_PROMPT
    
    # Agregar reglas específicas de anti-alucinación de Pablo
    system_prompt += (
        " los datos duros (costos, plazos, presencialidad, requisitos legales, fuentes oficiales) "
        " los debes tomar exclusivamente de la informacion entregada en el contexto, sin inventar "
        " ni modificar ninguno. nunca generes una fuente oficial (URL) que no sea la entregada. "
        " el usuario puede mencionar otras situaciones, tramites o documentos ademas del tramite "
        " principal indicado en el contexto. NO comentes, evalues ni des informacion sobre esas "
        " otras situaciones bajo ninguna circunstancia, ya que no tienes esa informacion verificada. "
        " concentra toda tu calidez y cercania unicamente en el tramite principal: puedes reconocer "
        " que este tramite es importante para el usuario, o transmitir cercania al explicarlo, "
        " pero sin mencionar la otra situacion que el usuario haya nombrado."
    )

    if ignorar_otros_tramites:
        system_prompt += (
            " el usuario menciono mas de un tramite en su pregunta. "
            " tu tarea es responder unicamente sobre el tramite indicado en el contexto, "
            " sin mencionar, comparar ni hacer referencia a ningun otro tramite."
        )

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

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS
    }

    input_tokens = estimate_tokens(json.dumps(payload["messages"], ensure_ascii=False))
    start = time.time()
    fallback = False
    error = None

    try:
        response = requests.post(LLM_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        respuesta = clean_model_text(raw_text)

        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", 0)
        else:
            output_tokens = estimate_tokens(respuesta)
    except Exception as e:
        respuesta = clean_model_text(item["respuesta"])
        input_tokens = 0
        output_tokens = 0
        fallback = True
        error = str(e)

    elapsed_ms = (time.time() - start) * 1000

    return respuesta, {
        "latency_ms": elapsed_ms,
        "ttft_ms": elapsed_ms * 0.35,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "fallback": fallback,
        "error": error
    }


def build_plan_text(ordered_items: list[dict]) -> str:
    lineas = []
    for i, item in enumerate(ordered_items):
        partes = [f"Paso {i + 1}: {item['titulo']}."]
        partes.append(clean_model_text(item["respuesta"]))
        if item.get("costo"):
            partes.append(f"Costo: {item['costo']}.")
        if item.get("presencialidad"):
            partes.append(f"Presencialidad: {item['presencialidad']}.")
        if item.get("requiere_clave_unica"):
            partes.append(f"Requiere ClaveUnica: {item['requiere_clave_unica']}.")
        lineas.append(" ".join(partes))
    return "\n\n".join(lineas)


def generate_plan_intro(user_query: str, ordered_items: list[dict]) -> Optional[str]:
    system_prompt = (
        "eres ARCO, un asistente para el registro civil y su orientacion. "
        "escribe una sola oracion breve introduciendo un plan de tramites. "
        "no menciones costos, plazos ni detalles, solo una introduccion general. "
        "maximo 20 palabras."
    )

    tramites_nombres = ", ".join(item["titulo"] for item in ordered_items)

    user_prompt = f"""
pregunta del usuario: {user_query}

tramites incluidos en el plan, en orden: {tramites_nombres}

escribe una sola oracion breve de introduccion al plan.
""".strip()

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 60
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        return clean_model_text(raw_text)
    except Exception:
        return None


def generate_aclaracion_intro(user_query: str, main_item: dict, concern_items: list[dict]) -> Optional[str]:
    system_prompt = (
        "eres ARCO, un asistente para el registro civil y su orientacion. "
        "escribe una sola oracion breve confirmando, segun el campo entregado, "
        "si el usuario necesita o no el documento que le preocupa para el tramite principal. "
        "no agregues costos, plazos ni pasos. maximo 20 palabras."
    )

    concern_titulos = " y ".join(item["titulo"] for item in concern_items)

    user_prompt = f"""
pregunta del usuario: {user_query}

tramite principal: {main_item['titulo']}
requiere clave unica: {main_item['requiere_clave_unica']}
documento que preocupa al usuario: {concern_titulos}

escribe una sola oracion breve confirmando si es necesario o no ese documento para este tramite.
""".strip()

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        return clean_model_text(raw_text)
    except Exception:
        return None


def build_aclaracion_text(main_item: dict, concern_items: list[dict]) -> str:
    concern_titulos = " ni tu ".join(item["titulo"] for item in concern_items)
    requiere = main_item["requiere_clave_unica"]

    if requiere == "no":
        aclaracion = f"No necesitas tu {concern_titulos} para este tramite."
    elif requiere == "si":
        aclaracion = f"Este tramite si requiere ClaveUnica vigente, y por lo tanto tambien tu {concern_titulos} al dia."
    else:
        aclaracion = f"La necesidad de tu {concern_titulos} depende de las condiciones especificas de tu caso."

    partes = [aclaracion, clean_model_text(main_item["respuesta"])]
    if main_item.get("costo"):
        partes.append(f"Costo: {main_item['costo']}.")
    partes.append(f"Fuente: {main_item['fuente']}.")
    return " ".join(partes)


def build_independent_responses(
    user_query: str,
    items: list[dict],
    history: list[ChatMessage]
) -> tuple[list[dict], int, int, Optional[float], bool]:
    resultados = []
    total_input = 0
    total_output = 0
    ttft_ms = None
    fallback = False
    for item in items:
        consulta_relevante = extract_relevant_clause(user_query, item)
        try:
            texto, meta = generate_llm_response(
                consulta_relevante, item, [], using_previous_context=False, ignorar_otros_tramites=True
            )
            total_input += meta.get("input_tokens", 0)
            total_output += meta.get("output_tokens", 0)
            if not ttft_ms and meta.get("ttft_ms"):
                ttft_ms = meta.get("ttft_ms")
            if meta.get("fallback"):
                fallback = True
            
            if response_has_foreign_url(texto, item["fuente"]):
                texto = build_single_item_fallback(item)
                fallback = True
        except Exception:
            texto = clean_model_text(item["respuesta"])
            fallback = True

        resultados.append({
            "tramite": item["titulo"],
            "respuesta": texto,
            "respuesta_base": item["respuesta"],
            "costo": item.get("costo", None),
            "duracion": item.get("duracion", None),
            "canal": item["canal"],
            "presencialidad": item["presencialidad"],
            "requiere_clave_unica": item["requiere_clave_unica"],
            "fuente": item["fuente"]
        })

    return resultados, total_input, total_output, ttft_ms, fallback


# ---------------------------------------------------------------------------
# Llamada al LLM
# ---------------------------------------------------------------------------

def call_llm(
    user_query: str,
    item: dict,
    history: list[ChatMessage],
    using_previous_context: bool = False,
) -> tuple[str, dict]:
    cfg = MODEL

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
        "model_label":   cfg["label"],
        "model_id":      cfg["model"],
    }


# ---------------------------------------------------------------------------
# WhatsApp Cloud API
# ---------------------------------------------------------------------------

def extract_whatsapp_message(payload: dict) -> Optional[tuple[str, str, bool]]:
    """
    Extrae el número del usuario y el texto recibido desde el payload de WhatsApp.

    Retorna:
      (from_number, text, True)  -> mensaje de texto normal del usuario
      (from_number, text, False) -> mensaje no textual; se responde sin pasar por LLM
      None                      -> evento sin mensaje útil, por ejemplo status updates
    """
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return None

        message = messages[0]
        from_number = message.get("from")
        message_type = message.get("type")

        if not from_number:
            return None

        if message_type != "text":
            return (
                from_number,
                "Por ahora ARCO solo puede responder mensajes de texto. Escribe tu consulta sobre trámites del Registro Civil.",
                False,
            )

        text = message.get("text", {}).get("body", "").strip()

        if not text:
            return None

        return from_number, text, True

    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("No se pudo extraer mensaje WhatsApp: %s", exc)
        return None


def build_whatsapp_reply(arco_response: dict) -> str:
    """
    Convierte la respuesta estructurada de ARCO en un mensaje compacto para WhatsApp.
    """
    modo = arco_response.get("modo")
    partes = ["ARCO - Orientación de trámite"]

    if modo == "plan":
        tramite_detectado = arco_response.get("tramite") or "Trámites"
        partes.append(f"Trámite detectado: {tramite_detectado}")
        intro = arco_response.get("intro") or "Aquí tienes el plan de trámites:"
        partes.append(f"\n{intro}")

        for step in arco_response.get("plan", []):
            partes.append("")
            nombre = step.get("tramite") or "Paso"
            partes.append(f"*{nombre}*")
            desc = step.get("respuesta") or step.get("respuesta_base") or ""
            if desc:
                partes.append(desc)
            partes.append(f"Costo: {step.get('costo') or 'No disponible'}")
            partes.append(f"Presencialidad: {step.get('presencialidad') or 'No disponible'}")
            partes.append(f"ClaveÚnica: {step.get('requiere_clave_unica') or 'No disponible'}")
            if step.get("fuente"):
                partes.append(f"Fuente: {step.get('fuente')}")

    elif modo == "consultas_independientes":
        tramite_detectado = arco_response.get("tramite") or "Consultas independientes"
        partes.append(f"Trámite detectado: {tramite_detectado}")
        
        for step in arco_response.get("consultas", []):
            partes.append("")
            nombre = step.get("tramite") or "Trámite"
            partes.append(f"*{nombre}*")
            desc = step.get("respuesta") or step.get("respuesta_base") or ""
            if desc:
                partes.append(desc)
            partes.append(f"Costo: {step.get('costo') or 'No disponible'}")
            partes.append(f"Presencialidad: {step.get('presencialidad') or 'No disponible'}")
            partes.append(f"ClaveÚnica: {step.get('requiere_clave_unica') or 'No disponible'}")
            if step.get("fuente"):
                partes.append(f"Fuente: {step.get('fuente')}")
    else:
        tramite = arco_response.get("tramite") or "Trámite no identificado"
        respuesta = arco_response.get("respuesta") or "No fue posible generar una respuesta."
        costo = arco_response.get("costo")
        duracion = arco_response.get("duracion")
        presencialidad = arco_response.get("presencialidad")
        clave_unica = arco_response.get("requiere_clave_unica")
        fuente = arco_response.get("fuente")

        partes.append(f"Trámite detectado: {tramite}\n")
        partes.append(respuesta)

        if costo:
            partes.append(f"\nCosto: {costo}")
        if duracion:
            partes.append(f"Duración: {duracion}")
        if presencialidad:
            partes.append(f"Presencialidad: {presencialidad}")
        if clave_unica:
            partes.append(f"ClaveÚnica: {clave_unica}")
        if fuente:
            partes.append(f"Fuente: {fuente}")

    partes.append("\nEsta orientación es informativa. Verifica siempre en canales oficiales.")
    return "\n".join(partes)[:3500]


def send_whatsapp_message(to_number: str, text: str) -> bool:
    """
    Envía un mensaje de texto usando WhatsApp Cloud API.
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.error("WhatsApp no configurado: falta WHATSAPP_TOKEN o WHATSAPP_PHONE_NUMBER_ID")
        return False

    url = (
        f"https://graph.facebook.com/{WHATSAPP_GRAPH_VERSION}/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {
            "preview_url": True,
            "body": text,
        },
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)

        if response.status_code >= 400:
            logger.error("Error enviando WhatsApp: %s - %s", response.status_code, response.text)
            return False

        logger.info("Mensaje WhatsApp enviado correctamente a %s", to_number)
        return True

    except requests.RequestException as exc:
        logger.error("Error de red enviando WhatsApp: %s", exc)
        return False


@app.get("/webhook/whatsapp")
def verify_whatsapp_webhook(request: Request):
    """
    Verificación inicial del webhook desde Meta.
    Meta llama esta ruta con hub.mode, hub.verify_token y hub.challenge.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN and challenge:
        logger.info("Webhook de WhatsApp verificado correctamente")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("Fallo verificación webhook WhatsApp")
    return PlainTextResponse(content="Token de verificación inválido", status_code=403)


@app.post("/webhook/whatsapp")
def receive_whatsapp_message(payload: dict):
    """
    Recibe mensajes entrantes desde WhatsApp, consulta ARCO y responde por WhatsApp.
    """
    logger.info("Webhook WhatsApp recibido")

    extracted = extract_whatsapp_message(payload)

    if not extracted:
        return {"status": "ignored", "reason": "payload sin mensaje de texto"}

    from_number, user_text, should_call_arco = extracted
    integrante = resolve_integrante(from_number)

    logger.info("Mensaje WhatsApp desde %s (%s): %s", from_number, integrante, user_text)

    if not should_call_arco:
        sent = send_whatsapp_message(from_number, user_text)
        return {
            "status": "processed",
            "sent": sent,
            "from": from_number,
            "type": "non_text",
        }

    question = Question(
        query=user_text,
        history=[],
        context=None,
        trace_id=f"whatsapp-{uuid.uuid4()}",
    )

    arco_response = ask_core(
        question,
        channel="whatsapp",
        integrante=integrante,
        telefono=from_number
    )
    whatsapp_text = build_whatsapp_reply(arco_response)
    sent = send_whatsapp_message(from_number, whatsapp_text)

    return {
        "status": "processed",
        "sent": sent,
        "from": from_number,
        "integrante": integrante,
        "tramite": arco_response.get("tramite"),
        "trace_id": arco_response.get("trace_id"),
    }


# Endpoints
# ---------------------------------------------------------------------------


@app.get("/privacy-policy", response_class=HTMLResponse)
def privacy_policy():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Política de Privacidad - ARCO TAVI</title>
    </head>
    <body>
        <h1>Política de Privacidad - ARCO TAVI</h1>

        <p>
            ARCO TAVI es un prototipo académico desarrollado para la asignatura
            Taller de Agentes Virtuales Inteligentes de la Universidad de Santiago de Chile.
        </p>

        <h2>Datos procesados</h2>
        <p>
            El sistema puede recibir mensajes de texto enviados por usuarios mediante WhatsApp
            con el objetivo de entregar orientación informativa sobre trámites del Registro Civil.
        </p>

        <h2>Uso de la información</h2>
        <p>
            Los mensajes se utilizan únicamente para generar una respuesta automática dentro del
            contexto del prototipo académico. No se venden datos personales ni se comparten con
            terceros para fines comerciales.
        </p>

        <h2>Registro de métricas</h2>
        <p>
            El sistema puede almacenar métricas técnicas como fecha de consulta, trámite detectado,
            modelo utilizado, latencia, cantidad de tokens y si hubo fallback. Estas métricas se usan
            solo para evaluación académica y mejora del prototipo.
        </p>

        <h2>Limitaciones</h2>
        <p>
            ARCO TAVI entrega orientación informativa. La información oficial debe verificarse
            siempre en los canales oficiales correspondientes.
        </p>

        <h2>Contacto</h2>
        <p>
            Para consultas sobre este prototipo, contactar al equipo desarrollador del proyecto ARCO TAVI.
        </p>
    </body>
    </html>
    """


@app.get("/")
def root():
    return {
        "message": "ARCO backend de producción",
        "modelo": MODEL["label"],
    }


@app.get("/models")
def get_models():
    return {"label": MODEL["label"], "url": MODEL["url"], "model_id": MODEL["model"]}


def ask_core(
    question: Question,
    channel: str = "web",
    integrante: Optional[str] = None,
    telefono: Optional[str] = None,
) -> dict:
    trace_id = question.trace_id or str(uuid.uuid4())
    query = normalize_text(question.query)

    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    fallback = False
    error = None
    using_previous_context = False

    # 1. Identificar todos los trámites posibles (RAG + Keywords)
    matched_ids = []
    seen_ids = set()

    # RAG matches (si está activo)
    if USE_RAG:
        rag_results = search_rag(question.query)
        for r in rag_results:
            rid = r.get("id")
            if rid and rid in ID_TO_ITEM and rid not in seen_ids:
                if has_keyword_overlap(question.query, ID_TO_ITEM[rid]):
                    matched_ids.append(rid)
                    seen_ids.add(rid)

    # Keyword matches
    kw_ids = keyword_match(query)
    for kid in kw_ids:
        if kid not in seen_ids:
            matched_ids.append(kid)
            seen_ids.add(kid)

    # Caso especial: licencia de conducir (fuera de dominio)
    query_raw_lower = question.query.lower()
    if not matched_ids and ("licencia de conducir" in query_raw_lower or "renovar licencia" in query_raw_lower):
        respuesta = (
            "La renovacion de licencia de conducir no corresponde al Registro Civil. "
            "ARCO esta enfocado en tramites del Registro Civil y su orientacion."
        )
        total_latency_ms = (time.time() - start_time) * 1000
        save_metric(base_metric_entry(
            trace_id, "Fuera del alcance", question.query, respuesta, channel, integrante, telefono,
            latency_ms=total_latency_ms,
            ttft_ms=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            fallback=False,
            error=None
        ))
        return {
            "tramite": "Fuera del alcance de ARCO", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": "https://www.chileatiende.gob.cl/fichas/20592-licencias-de-conducir",
            "trace_id": trace_id, "model_used": MODEL["label"],
        }

    # Seguimiento de contexto previo (solo si no hay coincidencias directas)
    context_item = None
    if not matched_ids and question.context and is_followup_query(question.query):
        context_item = context_to_item(question.context)
        if context_item:
            ctx_id = None
            for k_id, item in ID_TO_ITEM.items():
                if item["titulo"] == context_item["titulo"]:
                    ctx_id = k_id
                    break
            if ctx_id:
                matched_ids = [ctx_id]
                using_previous_context = True

    # Trámite no identificado
    if not matched_ids:
        respuesta = OFF_TOPIC_REFUSAL
        total_latency_ms = (time.time() - start_time) * 1000
        save_metric(base_metric_entry(
            trace_id, "Fuera del alcance", question.query, respuesta, channel, integrante, telefono,
            latency_ms=total_latency_ms,
            ttft_ms=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            fallback=False,
            error=None
        ))
        return {
            "tramite": "No identificado", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": None, "trace_id": trace_id,
            "model_used": MODEL["label"],
        }

    # Registro de log de MLOps
    logger.info(
        f"query='{question.query}' "
        f"canal={channel} "
        f"integrante={integrante or '-'} "
        f"matched_ids={matched_ids} "
        f"rag={USE_RAG} "
        f"followup={using_previous_context}"
    )

    # 2. Procesar flujo según cantidad de trámites identificados
    if len(matched_ids) == 1:
        matched_item = ID_TO_ITEM[matched_ids[0]]
        consulta_relevante = extract_relevant_clause(question.query, matched_item)
        unmatched_clauses = find_unmatched_clauses(question.query)

        try:
            respuesta_ia, meta = generate_llm_response(
                consulta_relevante,
                matched_item,
                question.history if using_previous_context else [],
                using_previous_context=using_previous_context
            )
            total_input_tokens += meta.get("input_tokens", 0)
            total_output_tokens += meta.get("output_tokens", 0)
            if meta.get("fallback"):
                fallback = True
            if meta.get("error"):
                error = meta["error"]

            if response_has_foreign_url(respuesta_ia, matched_item["fuente"]):
                respuesta_ia = build_single_item_fallback(matched_item)
                fallback = True
        except Exception as e:
            respuesta_ia = clean_model_text(matched_item["respuesta"])
            fallback = True
            error = str(e)

        if not fallback and normalize_text(respuesta_ia) == normalize_text(OFF_TOPIC_REFUSAL):
            total_latency_ms = (time.time() - start_time) * 1000
            ttft_ms = meta.get("ttft_ms") if meta else total_latency_ms * 0.35
            save_metric(base_metric_entry(
                trace_id, "Fuera del alcance", question.query, OFF_TOPIC_REFUSAL, channel, integrante, telefono,
                latency_ms=total_latency_ms,
                ttft_ms=ttft_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                fallback=False,
                error=None
            ))
            return {
                "tramite": "Fuera del alcance de ARCO",
                "respuesta": OFF_TOPIC_REFUSAL,
                "respuesta_base": None,
                "costo": None,
                "duracion": None,
                "canal": None,
                "presencialidad": None,
                "requiere_clave_unica": None,
                "fuente": None,
                "trace_id": trace_id,
                "model_used": MODEL["label"]
            }

        if unmatched_clauses:
            partes_sin_cubrir = "; ".join(unmatched_clauses)
            respuesta_ia = (
                f'Sobre "{partes_sin_cubrir}" no tengo informacion dentro del alcance de ARCO. '
                f"{respuesta_ia}"
            )

        total_latency_ms = (time.time() - start_time) * 1000
        ttft_ms = meta.get("ttft_ms") if (not fallback and meta) else total_latency_ms * 0.35
        save_metric(base_metric_entry(
            trace_id, matched_item["titulo"], question.query, respuesta_ia, channel, integrante, telefono,
            latency_ms=total_latency_ms,
            ttft_ms=ttft_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            fallback=fallback,
            error=error,
            using_previous_context=using_previous_context
        ))

        return {
            "tramite": matched_item["titulo"],
            "respuesta": respuesta_ia,
            "respuesta_base": matched_item["respuesta"],
            "costo": matched_item.get("costo", None),
            "duracion": matched_item.get("duracion", None),
            "canal": matched_item["canal"],
            "presencialidad": matched_item["presencialidad"],
            "requiere_clave_unica": matched_item["requiere_clave_unica"],
            "fuente": matched_item["fuente"],
            "trace_id": trace_id,
            "model_used": MODEL["label"]
        }

    # Más de un trámite identificado
    pares_relacionados = any(
        has_real_relation(matched_ids[i], matched_ids[j])
        for i in range(len(matched_ids))
        for j in range(i + 1, len(matched_ids))
    )

    if not pares_relacionados:
        concern_matches = [mid for mid in matched_ids if mid in CONCERN_IDS]
        main_matches = [mid for mid in matched_ids if mid not in CONCERN_IDS]

        if len(main_matches) == 1 and concern_matches:
            main_item = ID_TO_ITEM[main_matches[0]]
            concern_items = [ID_TO_ITEM[c] for c in concern_matches]

            respuesta_ia = build_aclaracion_text(main_item, concern_items)
            try:
                intro = generate_aclaracion_intro(question.query, main_item, concern_items)
                if intro:
                    respuesta_ia = f"{intro} {respuesta_ia}"
            except Exception as e:
                logger.warning(f"Error generando intro de aclaracion: {e}")

            total_latency_ms = (time.time() - start_time) * 1000
            ac_input = estimate_tokens(question.query) + 120
            ac_output = estimate_tokens(intro) if intro else 0
            save_metric(base_metric_entry(
                trace_id, main_item["titulo"], question.query, respuesta_ia, channel, integrante, telefono,
                latency_ms=total_latency_ms,
                ttft_ms=total_latency_ms * 0.35,
                input_tokens=ac_input,
                output_tokens=ac_output,
                total_tokens=ac_input + ac_output,
                fallback=False,
                error=None
            ))

            return {
                "modo": "aclaracion",
                "tramite": main_item["titulo"],
                "respuesta": respuesta_ia,
                "respuesta_base": main_item["respuesta"],
                "costo": main_item.get("costo", None),
                "duracion": main_item.get("duracion", None),
                "canal": main_item["canal"],
                "presencialidad": main_item["presencialidad"],
                "requiere_clave_unica": main_item["requiere_clave_unica"],
                "fuente": main_item["fuente"],
                "trace_id": trace_id,
                "model_used": MODEL["label"]
            }

        # Consultas independientes
        items_independientes = [ID_TO_ITEM[m] for m in matched_ids]
        consultas, ind_input, ind_output, first_ttft, ind_fallback = build_independent_responses(question.query, items_independientes, question.history)
        
        # Filtrar sub-consultas que resultaron ser off-topic
        consultas_filtradas = [c for c in consultas if normalize_text(c["respuesta"]) != normalize_text(OFF_TOPIC_REFUSAL)]
        if not consultas_filtradas:
            total_latency_ms = (time.time() - start_time) * 1000
            save_metric(base_metric_entry(
                trace_id, "Fuera del alcance", question.query, OFF_TOPIC_REFUSAL, channel, integrante, telefono,
                latency_ms=total_latency_ms,
                ttft_ms=first_ttft if first_ttft else total_latency_ms * 0.35,
                input_tokens=ind_input,
                output_tokens=ind_output,
                total_tokens=ind_input + ind_output,
                fallback=ind_fallback,
                error=None
            ))
            return {
                "tramite": "Fuera del alcance de ARCO",
                "respuesta": OFF_TOPIC_REFUSAL,
                "respuesta_base": None,
                "costo": None,
                "duracion": None,
                "canal": None,
                "presencialidad": None,
                "requiere_clave_unica": None,
                "fuente": None,
                "trace_id": trace_id,
                "model_used": MODEL["label"],
                "fallback": ind_fallback
            }
        
        consultas = consultas_filtradas
        texto_combinado = "\n\n".join(f"Sobre {c['tramite']}: {c['respuesta']}" for c in consultas)

        total_latency_ms = (time.time() - start_time) * 1000
        ttft_ms = first_ttft if first_ttft else total_latency_ms * 0.35
        save_metric(base_metric_entry(
            trace_id, " + ".join(c["tramite"] for c in consultas), question.query, texto_combinado, channel, integrante, telefono,
            latency_ms=total_latency_ms,
            ttft_ms=ttft_ms,
            input_tokens=ind_input,
            output_tokens=ind_output,
            total_tokens=ind_input + ind_output,
            fallback=ind_fallback,
            error=None
        ))

        return {
            "modo": "consultas_independientes",
            "tramite": " + ".join(c["tramite"] for c in consultas),
            "respuesta": texto_combinado,
            "respuesta_base": None,
            "costo": None,
            "duracion": None,
            "canal": None,
            "presencialidad": None,
            "requiere_clave_unica": None,
            "fuente": None,
            "consultas": consultas,
            "trace_id": trace_id,
            "model_used": MODEL["label"],
            "fallback": ind_fallback
        }

    # Plan estructurado (Trámites relacionados con dependencias)
    expanded_ids = expand_dependencies(matched_ids)
    ordered_ids = topological_order(expanded_ids)
    ordered_items = [ID_TO_ITEM[i] for i in ordered_ids if i in ID_TO_ITEM]

    intro = None
    try:
        intro = generate_plan_intro(question.query, ordered_items)
    except Exception as e:
        logger.warning(f"Error generando intro de plan: {e}")

    plan_text = build_plan_text(ordered_items)
    respuesta_ia = f"{intro}\n\n{plan_text}" if intro else plan_text

    total_latency_ms = (time.time() - start_time) * 1000
    plan_input = estimate_tokens(question.query) + 150
    plan_output = estimate_tokens(intro) if intro else 0
    save_metric(base_metric_entry(
        trace_id, " + ".join(item["titulo"] for item in ordered_items), question.query, respuesta_ia, channel, integrante, telefono,
        latency_ms=total_latency_ms,
        ttft_ms=total_latency_ms * 0.35,
        input_tokens=plan_input,
        output_tokens=plan_output,
        total_tokens=plan_input + plan_output,
        fallback=False,
        error=None
    ))

    return {
        "modo": "plan",
        "tramite": " + ".join(item["titulo"] for item in ordered_items),
        "respuesta": respuesta_ia,
        "intro": intro,
        "respuesta_base": None,
        "costo": None,
        "duracion": None,
        "canal": None,
        "presencialidad": None,
        "requiere_clave_unica": None,
        "fuente": None,
        "plan": [
            {
                "orden": i + 1,
                "tramite": item["titulo"],
                "respuesta_base": item["respuesta"],
                "costo": item.get("costo", None),
                "duracion": item.get("duracion", None),
                "canal": item["canal"],
                "presencialidad": item["presencialidad"],
                "requiere_clave_unica": item["requiere_clave_unica"],
                "fuente": item["fuente"]
            }
            for i, item in enumerate(ordered_items)
        ],
        "trace_id": trace_id,
        "model_used": MODEL["label"]
    }


@app.post("/ask")
def ask_question(question: Question):
    return ask_core(question, channel="web")


@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    save_feedback_entry({
        "trace_id":  feedback.trace_id,
        "score":     feedback.score,
        "comment":   feedback.comment,
        "timestamp": datetime.now().isoformat(),
    })
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Endpoints de métricas
# ---------------------------------------------------------------------------

@app.get("/metrics")
def get_metrics():
    metrics   = load_metrics()
    feedbacks = load_feedbacks()
    fb_index  = {fb["trace_id"]: fb for fb in feedbacks}
    for m in metrics:
        m["feedback"] = fb_index.get(m["trace_id"])
    return metrics


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0
    s = sorted(values)
    idx = max(int(len(s) * 0.95) - 1, 0)
    return s[idx]


NO_MATCH_TRAMITES = {"No identificado", "Fuera del alcance"}


def _summarize(subset: list[dict], fb_index: dict) -> dict:
    if not subset:
        return {
            "total_consultas": 0, "promedio_latencia_ms": 0, "p95_latencia_ms": 0,
            "promedio_ttft_ms": 0, "p95_ttft_ms": 0, "total_tokens": 0,
            "tokens_por_consulta": 0, "tokens_por_segundo": 0, "tasa_fallback": 0,
            "tasa_no_identificado": 0, "consultas_por_hora": 0,
            "feedback_positivos": 0, "feedback_negativos": 0, "feedback_neutral": 0,
            "pct_feedback_positivo": 0,
        }

    latencies = [m["latency_ms"] for m in subset if m.get("latency_ms")]
    ttfts     = [m["ttft_ms"] for m in subset if m.get("ttft_ms")]
    total_tok = sum(m.get("total_tokens", 0) for m in subset)
    fallbacks = sum(1 for m in subset if m.get("fallback"))
    no_match  = sum(1 for m in subset if m.get("tramite") in NO_MATCH_TRAMITES)

    throughputs = []
    for m in subset:
        gen_ms = (m.get("latency_ms") or 0) - (m.get("ttft_ms") or 0)
        out_tok = m.get("output_tokens", 0)
        if gen_ms > 0 and out_tok > 0:
            throughputs.append(out_tok / (gen_ms / 1000))

    timestamps = sorted(m["timestamp"] for m in subset if m.get("timestamp"))
    if len(timestamps) >= 2:
        span_hours = max(
            (datetime.fromisoformat(timestamps[-1]) - datetime.fromisoformat(timestamps[0])).total_seconds() / 3600,
            1 / 60,
        )
        consultas_por_hora = round(len(subset) / span_hours, 2)
    else:
        consultas_por_hora = 0

    fb_subset = [fb_index[m["trace_id"]] for m in subset if m["trace_id"] in fb_index]
    positive  = sum(1 for fb in fb_subset if fb["score"] >= 0.8)
    negative  = sum(1 for fb in fb_subset if fb["score"] <= 0.2)
    neutral   = len(fb_subset) - positive - negative

    return {
        "total_consultas":       len(subset),
        "promedio_latencia_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p95_latencia_ms":       round(_percentile95(latencies), 1),
        "promedio_ttft_ms":      round(sum(ttfts) / len(ttfts), 1) if ttfts else 0,
        "p95_ttft_ms":           round(_percentile95(ttfts), 1),
        "total_tokens":          total_tok,
        "tokens_por_consulta":   round(total_tok / len(subset), 1),
        "tokens_por_segundo":    round(sum(throughputs) / len(throughputs), 1) if throughputs else 0,
        "tasa_fallback":         round(fallbacks / len(subset) * 100, 2),
        "tasa_no_identificado":  round(no_match / len(subset) * 100, 2),
        "consultas_por_hora":    consultas_por_hora,
        "feedback_positivos":    positive,
        "feedback_negativos":    negative,
        "feedback_neutral":      neutral,
        "pct_feedback_positivo": round(positive / len(fb_subset) * 100, 1) if fb_subset else 0,
    }


def get_memory_usage_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)


def load_eval_report() -> Optional[dict]:
    if not EVAL_REPORT_PATH.exists():
        return None
    with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/stats")
def get_stats():
    metrics = load_metrics()
    if not metrics:
        return {"error": "No hay métricas aún"}

    feedbacks = load_feedbacks()
    fb_index  = {fb["trace_id"]: fb for fb in feedbacks}

    resumen = _summarize(metrics, fb_index)
    resumen["modelo_activo"]      = MODEL["label"]
    resumen["consultas_web"]      = sum(1 for m in metrics if m.get("channel") == "web")
    resumen["consultas_whatsapp"] = sum(1 for m in metrics if m.get("channel") == "whatsapp")
    resumen["memoria_mb"]         = get_memory_usage_mb()

    eval_report = load_eval_report()
    if eval_report:
        resumen["eval_accuracy"]         = round(eval_report.get("accuracy", 0) * 100, 1)
        resumen["eval_correctos"]        = eval_report.get("correctos")
        resumen["eval_total_casos"]      = eval_report.get("total_casos")
        resumen["eval_aprobado"]         = eval_report.get("aprobado")
        resumen["eval_accuracy_minima"]  = round(eval_report.get("accuracy_minima", 0) * 100, 1)
    else:
        resumen["eval_accuracy"] = None

    return resumen


@app.get("/eval/latest")
def get_eval_latest():
    report = load_eval_report()
    if not report:
        return {"error": "No hay reporte de evaluación aún. Corre python eval/evaluador.py"}
    return report


@app.get("/stats/whatsapp")
def get_stats_whatsapp():
    metrics   = [m for m in load_metrics() if m.get("channel") == "whatsapp"]
    feedbacks = load_feedbacks()
    fb_index  = {fb["trace_id"]: fb for fb in feedbacks}

    por_integrante: dict[str, list[dict]] = {}
    for m in metrics:
        nombre = m.get("integrante") or "Desconocido"
        por_integrante.setdefault(nombre, []).append(m)

    result = {}
    for nombre, subset in por_integrante.items():
        s = _summarize(subset, fb_index)
        s["ultima_consulta"] = max(m["timestamp"] for m in subset)
        result[nombre] = s
    return result


@app.get("/metrics/timeline")
def get_timeline(limit: int = 50):
    metrics = sorted(load_metrics(), key=lambda x: x.get("timestamp", ""))
    subset  = metrics[-limit:]
    return [
        {
            "timestamp":  m["timestamp"],
            "latency_ms": round(m.get("latency_ms", 0), 1),
            "ttft_ms":    round(m["ttft_ms"], 1) if m.get("ttft_ms") else None,
            "tokens":     m.get("total_tokens", 0),
            "fallback":   m.get("fallback", False),
            "tramite":    m.get("tramite", ""),
            "channel":    m.get("channel", "web"),
            "integrante": m.get("integrante"),
        }
        for m in subset
    ]



