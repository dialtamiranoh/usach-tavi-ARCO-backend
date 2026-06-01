from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import json
import unicodedata
import requests
import re

from dotenv import load_dotenv
import os



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

# LLM_URL = "http://127.0.0.1:8001/v1/chat/completions"
# LLM_MODEL = "arco-llm"


load_dotenv()

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "arco-llm")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "140"))


class ChatMessage(BaseModel):
    role: str
    content: str


class ContextData(BaseModel):
    tramite: Optional[str] = None
    respuesta_base: Optional[str] = None
    costo: Optional[str] = None
    duracion: Optional[str] = None
    canal: Optional[str] = None
    presencialidad: Optional[str] = None
    requiere_clave_unica: Optional[str] = None
    fuente: Optional[str] = None


class Question(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    context: Optional[ContextData] = None


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text


def clean_model_text(text: str) -> str:
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace('"', "")
    text = text.replace("•", " ")
    text = text.replace("*", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_followup_query(query: str) -> bool:
    query = normalize_text(query)

    followup_hints = [
        "y eso",
        "eso",
        "ese tramite",
        "esa gestion",
        "ese documento",
        "se puede",
        "requiere",
        "necesita",
        "necesito",
        "clave unica",
        "presencial",
        "online",
        "en linea",
        "por internet",
        "cuanto",
        "cuesta",
        "costo",
        "donde",
        "como",
        "cuando",
        "que necesito",
        "que documentos",
        "y si",
        "y para eso"
    ]

    return any(hint in query for hint in followup_hints)


def context_to_item(context: ContextData) -> Optional[dict]:
    if not context:
        return None

    if not context.tramite or not context.respuesta_base:
        return None

    return {
        "titulo": context.tramite,
        "respuesta": context.respuesta_base,
        "costo": context.costo,
        "duracion": context.duracion,
        "canal": context.canal,
        "presencialidad": context.presencialidad,
        "requiere_clave_unica": context.requiere_clave_unica,
        "fuente": context.fuente,
    }


def build_history_text(history: list[ChatMessage]) -> str:
    if not history:
        return "sin historial previo"

    lines = []
    for message in history[-6:]:
        role = message.role.strip().lower()
        content = message.content.strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def generate_llm_response(
    user_query: str,
    item: dict,
    history: list[ChatMessage],
    using_previous_context: bool = False
) -> str:
    system_prompt = (
        "eres ARCO, un asistente para el registro civil y su orientacion. "
        "responde solo con la informacion entregada. "
        "no inventes requisitos, costos, plazos ni pasos. "
        "responde en espanol claro, breve y natural. "
        "usa un solo parrafo, sin listas y con maximo 3 oraciones. "
        "si la pregunta es de seguimiento, responde considerando que el usuario sigue hablando del mismo tramite. "
        "menciona claramente el costo del trámite si está disponible. "
        "menciona si requiere presencialidad, si requiere clave unica y termina con la fuente oficial."
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
        # "temperature": 0.1,
        # "max_tokens": 140
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS
    }

    response = requests.post(LLM_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    raw_text = data["choices"][0]["message"]["content"].strip()
    return clean_model_text(raw_text)


@app.get("/")
def read_root():
    return {"message": "ARCO backend funcionando con LLM local"}


@app.post("/ask")
def ask_question(question: Question):
    query = normalize_text(question.query)

    matched_item = None
    using_previous_context = False

    for item in KNOWLEDGE:
        for keyword in item["keywords"]:
            if normalize_text(keyword) in query:
                matched_item = item
                break
        if matched_item:
            break

    if "licencia de conducir" in query or "renovar licencia" in query:
        return {
            "tramite": "Fuera del alcance de ARCO",
            "respuesta": "La renovacion de licencia de conducir no corresponde al Registro Civil. ARCO esta enfocado en tramites del Registro Civil y su orientacion.",
            "respuesta_base": None,
            "costo": None,
            "duracion": None,
            "canal": None,
            "presencialidad": None,
            "requiere_clave_unica": None,
            "fuente": "https://www.chileatiende.gob.cl/fichas/20592-licencias-de-conducir"
        }

    if not matched_item and question.context and is_followup_query(question.query):
        context_item = context_to_item(question.context)
        if context_item:
            matched_item = context_item
            using_previous_context = True

    if matched_item:
        try:
            respuesta_ia = generate_llm_response(
                question.query,
                matched_item,
                question.history,
                using_previous_context=using_previous_context
            )
        except Exception:
            respuesta_ia = clean_model_text(matched_item["respuesta"])

        return {
            "tramite": matched_item["titulo"],
            "respuesta": respuesta_ia,
            "respuesta_base": matched_item["respuesta"],
            "costo": matched_item.get("costo", None),
            "duracion": matched_item.get("duracion", None),
            "canal": matched_item["canal"],
            "presencialidad": matched_item["presencialidad"],
            "requiere_clave_unica": matched_item["requiere_clave_unica"],
            "fuente": matched_item["fuente"]
        }

    return {
        "tramite": "No identificado",
        "respuesta": "ARCO todavia no tiene informacion suficiente para orientar ese tramite dentro del alcance actual del demo.",
        "costo": None,
        "respuesta_base": None,
        "duracion": None,
        "canal": None,
        "presencialidad": None,
        "requiere_clave_unica": None,
        "fuente": None
    }