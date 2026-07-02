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

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

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
    tramite = arco_response.get("tramite") or "Trámite no identificado"
    respuesta = arco_response.get("respuesta") or "No fue posible generar una respuesta."
    costo = arco_response.get("costo")
    duracion = arco_response.get("duracion")
    fuente = arco_response.get("fuente")
    modelo = arco_response.get("model_used")

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

    if modelo:
        partes.append(f"Modelo usado: {modelo}")

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

    logger.info("Mensaje WhatsApp desde %s: %s", from_number, user_text)

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

    arco_response = ask_question(question)
    whatsapp_text = build_whatsapp_reply(arco_response)
    sent = send_whatsapp_message(from_number, whatsapp_text)

    return {
        "status": "processed",
        "sent": sent,
        "from": from_number,
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


@app.post("/ask")
def ask_question(question: Question):
    trace_id  = question.trace_id or str(uuid.uuid4())
    query     = normalize_text(question.query)

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
        return {
            "tramite": "Fuera del alcance de ARCO", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": "https://www.chileatiende.gob.cl/fichas/20592-licencias-de-conducir",
            "trace_id": trace_id, "model_used": MODEL["label"],
        }

    # Seguimiento de contexto previo
    if not matched_item and question.context and is_followup_query(question.query):
        ctx_item = context_to_item(question.context)
        if ctx_item:
            matched_item           = ctx_item
            using_previous_context = True

    logger.info(
        f"query='{question.query}' "
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
        return {
            "tramite": "No identificado", "respuesta": respuesta,
            "respuesta_base": None, "costo": None, "duracion": None,
            "canal": None, "presencialidad": None, "requiere_clave_unica": None,
            "fuente": None, "trace_id": trace_id,
            "model_used": MODEL["label"],
        }

    # Llamada al LLM
    try:
        respuesta_ia, meta = call_llm(
            question.query, matched_item, question.history,
            using_previous_context,
        )
    except Exception as e:
        respuesta_ia = clean_model_text(matched_item["respuesta"])
        meta = {
            "latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "fallback": True, "error": str(e),
            "model_label": MODEL["label"], "model_id": MODEL["model"],
        }

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
        "model_used":            MODEL["label"],
        "latency_ms":            round(meta["latency_ms"], 0),
        "total_tokens":          meta["total_tokens"],
        "fallback":              meta["fallback"],
    }


