"""
main.py — Backend unificado ARCO
Ejecutar:  python main.py   (o  uvicorn main:app --host 0.0.0.0 --port 8000 --reload)
Nota WSL: no omitas --host 0.0.0.0, si no uvicorn queda inalcanzable desde Windows.

Modelo LLM:
  Se carga automáticamente in-process (llama-cpp-python) desde la carpeta
  de modelos (por defecto ../modelos). No requiere levantar un servidor
  LLM aparte: al iniciar main.py el modelo ya queda listo para responder.

Archivos de persistencia:
  benchmark_metrics.jsonl  — una línea por consulta (canal web o whatsapp)
  benchmark_feedback.jsonl — feedback por trace_id
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
import re
import time
import uuid
import threading
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
logging.getLogger("llama_cpp").setLevel(logging.WARNING)

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
# Configuración del modelo único (carga automática desde carpeta modelos/)
# ---------------------------------------------------------------------------
MODEL_DIR             = Path(os.getenv("MODEL_DIR", str(BASE_DIR.parent / "modelos")))
MODEL_PATH_OVERRIDE   = os.getenv("MODEL_PATH", "")
# qwen es la preferencia por defecto: granite-4.0-1b usa una arquitectura híbrida
# Mamba2+atención que llama-cpp-python aún no soporta correctamente (genera texto
# incoherente). Qwen2.5-3B usa una arquitectura transformer estándar y funciona bien.
PREFERRED_MODEL_HINT  = os.getenv("MODEL_NAME", "qwen")
LLM_N_CTX             = int(os.getenv("LLM_N_CTX", "2048"))
LLM_N_THREADS         = int(os.getenv("LLM_N_THREADS", str(os.cpu_count() or 4)))
LLM_TEMPERATURE       = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS        = int(os.getenv("LLM_MAX_TOKENS",  "140"))

# Permite a los tests / pipelines saltarse la carga del modelo (lento y pesado).
SKIP_MODEL_LOAD = os.getenv("ARCO_SKIP_MODEL_LOAD", "false").lower() == "true"


def discover_model_path() -> Path:
    if MODEL_PATH_OVERRIDE:
        p = Path(MODEL_PATH_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"MODEL_PATH configurado pero no existe: {p}")
        return p

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta de modelos '{MODEL_DIR}'. "
            f"Configura MODEL_DIR o MODEL_PATH en el archivo .env."
        )

    candidates = sorted(MODEL_DIR.glob("**/*.gguf"))
    if not candidates:
        raise FileNotFoundError(f"No se encontró ningún archivo .gguf dentro de '{MODEL_DIR}'.")

    preferred = [c for c in candidates if PREFERRED_MODEL_HINT.lower() in str(c).lower()]
    return preferred[0] if preferred else candidates[0]


def infer_model_label(model_path: Path) -> str:
    name = model_path.stem.lower()
    if "granite" in name:
        return "Granite-4.0-1B"
    if "qwen" in name:
        return "Qwen2.5-3B"
    return model_path.stem


_llm_lock: threading.Lock = threading.Lock()
LLM = None
MODEL_PATH = None
MODEL_LABEL = os.getenv("MODEL_LABEL", "")

if not SKIP_MODEL_LOAD:
    from llama_cpp import Llama

    MODEL_PATH  = discover_model_path()
    MODEL_LABEL = MODEL_LABEL or infer_model_label(MODEL_PATH)

    logger.info(f"Cargando modelo LLM desde '{MODEL_PATH}' ...")
    _t0 = time.time()
    LLM = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        verbose=False,
    )
    logger.info(f"Modelo '{MODEL_LABEL}' cargado en {time.time() - _t0:.1f}s")
else:
    MODEL_LABEL = MODEL_LABEL or "Qwen2.5-3B"
    logger.info("Carga del modelo LLM omitida (ARCO_SKIP_MODEL_LOAD=true)")

# ---------------------------------------------------------------------------
# Configuración WhatsApp Cloud API
# ---------------------------------------------------------------------------
import requests

WHATSAPP_TOKEN            = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID  = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN     = os.getenv("WHATSAPP_VERIFY_TOKEN", "arco_tavi_verify_token")
WHATSAPP_GRAPH_VERSION    = os.getenv("WHATSAPP_GRAPH_VERSION", "v22.0")

# Mapeo "Nombre:numero,Nombre:numero" -> para mostrar consultas por integrante en el dashboard.
WHATSAPP_TEAM: dict[str, str] = {}
for _pair in os.getenv("WHATSAPP_TEAM", "").split(","):
    _pair = _pair.strip()
    if not _pair or ":" not in _pair:
        continue
    _name, _phone = _pair.split(":", 1)
    WHATSAPP_TEAM[_phone.strip()] = _name.strip()


def resolve_integrante(phone_number: str) -> str:
    return WHATSAPP_TEAM.get(phone_number, phone_number)


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
    history:  list[ChatMessage] = Field(default_factory=list)
    context:  Optional[ContextData] = None
    trace_id: Optional[str] = None


class Feedback(BaseModel):
    trace_id: str
    model:    Optional[str] = None
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


def base_metric_entry(
    trace_id: str, tramite: str, user_query: str, respuesta: str,
    channel: str, integrante: Optional[str], telefono: Optional[str] = None, **extra,
) -> dict:
    entry = {
        "trace_id":               trace_id,
        "timestamp":              datetime.now().isoformat(),
        "tramite":                tramite,
        "user_query":             user_query,
        "respuesta":              respuesta,
        "channel":                channel,
        "integrante":             integrante,
        "telefono":               telefono,
        "model_label":            MODEL_LABEL,
        "latency_ms":             0,
        "ttft_ms":                None,
        "input_tokens":           0,
        "output_tokens":          0,
        "total_tokens":           0,
        "fallback":               False,
        "error":                  None,
        "using_previous_context": False,
    }
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# Llamada al LLM (in-process, con streaming para medir time-to-first-token)
# ---------------------------------------------------------------------------

def call_llm(
    user_query: str,
    item: dict,
    history: list[ChatMessage],
    using_previous_context: bool = False,
) -> tuple[str, dict]:
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

    start    = time.time()
    ttft_ms  = None
    fallback = False
    error    = None

    if LLM is None:
        respuesta     = clean_model_text(item["respuesta"])
        input_tokens  = 0
        output_tokens = 0
        fallback      = True
        error         = "Modelo LLM no cargado"
    else:
        try:
            with _llm_lock:
                stream = LLM.create_chat_completion(
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    stream=True,
                )
                pieces = []
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (time.time() - start) * 1000
                        pieces.append(piece)

                raw_text = "".join(pieces).strip()
                respuesta = clean_model_text(raw_text)

                if ttft_ms is None:
                    ttft_ms = (time.time() - start) * 1000

                prompt_text   = system_prompt + "\n" + user_prompt
                input_tokens  = len(LLM.tokenize(prompt_text.encode("utf-8")))
                output_tokens = len(LLM.tokenize(raw_text.encode("utf-8"), add_bos=False)) if raw_text else 0

        except Exception as e:
            respuesta     = clean_model_text(item["respuesta"])
            input_tokens  = 0
            output_tokens = 0
            fallback      = True
            error         = str(e)
            ttft_ms       = None

    elapsed_ms = (time.time() - start) * 1000

    return respuesta, {
        "latency_ms":    elapsed_ms,
        "ttft_ms":       ttft_ms,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "fallback":      fallback,
        "error":         error,
    }


# ---------------------------------------------------------------------------
# Núcleo de la lógica de /ask (compartido entre web y WhatsApp)
# ---------------------------------------------------------------------------

def ask_core(
    *,
    query: str,
    history: list[ChatMessage],
    context: Optional[ContextData],
    trace_id: Optional[str],
    channel: str,
    integrante: Optional[str],
    telefono: Optional[str] = None,
) -> dict:
    trace_id     = trace_id or str(uuid.uuid4())
    query_norm   = normalize_text(query)

    matched_item           = None
    using_previous_context = False

    # RAG (si está activo)
    if USE_RAG:
        rag_results = search_rag(query)
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
                if normalize_text(kw) in query_norm:
                    matched_item = item
                    break
            if matched_item:
                break

    # Caso especial: licencia de conducir
    if "licencia de conducir" in query_norm or "renovar licencia" in query_norm:
        respuesta = (
            "La renovacion de licencia de conducir no corresponde al Registro Civil. "
            "ARCO esta enfocado en tramites del Registro Civil y su orientacion."
        )
        save_metric(base_metric_entry(
            trace_id, "Fuera del alcance", query, respuesta, channel, integrante, telefono,
        ))
        return {
            "tramite": "Fuera del alcance de ARCO", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": "https://www.chileatiende.gob.cl/fichas/20592-licencias-de-conducir",
            "trace_id": trace_id, "model_used": MODEL_LABEL,
        }

    # Seguimiento de contexto previo
    if not matched_item and context and is_followup_query(query):
        ctx_item = context_to_item(context)
        if ctx_item:
            matched_item           = ctx_item
            using_previous_context = True

    logger.info(
        f"query='{query}' "
        f"canal={channel} "
        f"integrante={integrante or '-'} "
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
        save_metric(base_metric_entry(
            trace_id, "No identificado", query, respuesta, channel, integrante, telefono,
        ))
        return {
            "tramite": "No identificado", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": None, "trace_id": trace_id,
            "model_used": MODEL_LABEL,
        }

    # Llamada al LLM
    try:
        respuesta_ia, meta = call_llm(query, matched_item, history, using_previous_context)
    except Exception as e:
        respuesta_ia = clean_model_text(matched_item["respuesta"])
        meta = {
            "latency_ms": 0, "ttft_ms": None, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "fallback": True, "error": str(e),
        }

    save_metric(base_metric_entry(
        trace_id, matched_item["titulo"], query, respuesta_ia, channel, integrante, telefono,
        latency_ms=meta.get("latency_ms", 0),
        ttft_ms=meta.get("ttft_ms"),
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
        total_tokens=meta.get("total_tokens", 0),
        fallback=meta.get("fallback", False),
        error=meta.get("error"),
        using_previous_context=using_previous_context,
    ))

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
        "model_used":            MODEL_LABEL,
        "latency_ms":            round(meta.get("latency_ms", 0), 0),
        "ttft_ms":               round(meta["ttft_ms"], 0) if meta.get("ttft_ms") else None,
        "total_tokens":          meta.get("total_tokens", 0),
        "fallback":              meta.get("fallback", False),
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
    tramite = arco_response.get("tramite") or "Trámite no identificado"
    respuesta = arco_response.get("respuesta") or "No fue posible generar una respuesta."
    costo = arco_response.get("costo")
    duracion = arco_response.get("duracion")
    fuente = arco_response.get("fuente")

    partes = [
        "ARCO - Orientación de trámite",
        f"Trámite detectado: {tramite}",
        "",
        respuesta,
    ]

    if costo:
        partes.append(f"\nCosto: {costo}")

    if duracion:
        partes.append(f"Duración: {duracion}")

    if fuente:
        partes.append(f"Fuente: {fuente}")

    partes.append("\nEsta orientación es informativa. Verifica siempre en canales oficiales.")

    texto = "\n".join(partes)

    # Para demo conviene mantenerlo compacto.
    return texto[:3500]


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

    arco_response = ask_core(
        query=user_text,
        history=[],
        context=None,
        trace_id=f"whatsapp-{uuid.uuid4()}",
        channel="whatsapp",
        integrante=integrante,
        telefono=from_number,
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


# ---------------------------------------------------------------------------
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
            canal utilizado, latencia, cantidad de tokens y si hubo fallback. Estas métricas se usan
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
        "message": "ARCO backend unificado",
        "modelo": {
            "label": MODEL_LABEL,
            "path": str(MODEL_PATH) if MODEL_PATH else None,
            "cargado": LLM is not None,
        },
    }


@app.get("/models")
def get_models():
    return {
        "label":   MODEL_LABEL,
        "path":    str(MODEL_PATH) if MODEL_PATH else None,
        "cargado": LLM is not None,
    }


@app.post("/ask")
def ask_question(question: Question):
    return ask_core(
        query=question.query,
        history=question.history,
        context=question.context,
        trace_id=question.trace_id,
        channel="web",
        integrante=None,
    )


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
            "tasa_no_identificado": 0,
            "feedback_positivos": 0, "feedback_negativos": 0, "feedback_neutral": 0,
            "pct_feedback_positivo": 0,
        }

    latencies = [m["latency_ms"] for m in subset if m.get("latency_ms")]
    ttfts     = [m["ttft_ms"] for m in subset if m.get("ttft_ms")]
    total_tok = sum(m.get("total_tokens", 0) for m in subset)
    fallbacks = sum(1 for m in subset if m.get("fallback"))
    no_match  = sum(1 for m in subset if m.get("tramite") in NO_MATCH_TRAMITES)

    # tokens/seg de generación pura: excluye el tiempo de prefill (TTFT), solo el tramo de salida.
    throughputs = []
    for m in subset:
        gen_ms = (m.get("latency_ms") or 0) - (m.get("ttft_ms") or 0)
        out_tok = m.get("output_tokens", 0)
        if gen_ms > 0 and out_tok > 0:
            throughputs.append(out_tok / (gen_ms / 1000))

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
        "feedback_positivos":    positive,
        "feedback_negativos":    negative,
        "feedback_neutral":      neutral,
        "pct_feedback_positivo": round(positive / len(fb_subset) * 100, 1) if fb_subset else 0,
    }


@app.get("/stats")
def get_stats():
    metrics = load_metrics()
    if not metrics:
        return {"error": "No hay métricas aún"}

    feedbacks = load_feedbacks()
    fb_index  = {fb["trace_id"]: fb for fb in feedbacks}

    resumen = _summarize(metrics, fb_index)
    resumen["modelo_activo"]      = MODEL_LABEL
    resumen["consultas_web"]      = sum(1 for m in metrics if m.get("channel") == "web")
    resumen["consultas_whatsapp"] = sum(1 for m in metrics if m.get("channel") == "whatsapp")
    return resumen



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


# ---------------------------------------------------------------------------
# Ejecución directa: `python main.py` levanta el backend y el modelo se
# carga automáticamente al importar este módulo (ver bloque de carga arriba).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
