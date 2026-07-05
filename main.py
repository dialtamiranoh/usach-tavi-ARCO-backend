from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import json
import unicodedata
import requests
import re

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

LLM_URL = "http://127.0.0.1:8001/v1/chat/completions"
LLM_MODEL = "arco-llm"


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
        "temperature": 0.1,
        "max_tokens": 140
    }

    response = requests.post(LLM_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    raw_text = data["choices"][0]["message"]["content"].strip()
    return clean_model_text(raw_text)


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
) -> list[dict]:
    resultados = []
    for item in items:
        try:
            texto = generate_llm_response(user_query, item, [], using_previous_context=False)
        except Exception:
            texto = clean_model_text(item["respuesta"])

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

    return resultados


@app.post("/ask")
def ask_question(question: Question):
    query = normalize_text(question.query)

    matched_ids = keyword_match(query)

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

    if not matched_ids and question.context and is_followup_query(question.query):
        context_item = context_to_item(question.context)
        if context_item:
            try:
                respuesta_ia = generate_llm_response(
                    question.query,
                    context_item,
                    question.history,
                    using_previous_context=True
                )
            except Exception:
                respuesta_ia = clean_model_text(context_item["respuesta"])

            return {
                "tramite": context_item["titulo"],
                "respuesta": respuesta_ia,
                "respuesta_base": context_item["respuesta"],
                "costo": context_item.get("costo", None),
                "duracion": context_item.get("duracion", None),
                "canal": context_item["canal"],
                "presencialidad": context_item["presencialidad"],
                "requiere_clave_unica": context_item["requiere_clave_unica"],
                "fuente": context_item["fuente"]
            }

    if matched_ids:
        if len(matched_ids) == 1:
            matched_item = ID_TO_ITEM[matched_ids[0]]
            try:
                respuesta_ia = generate_llm_response(
                    question.query,
                    matched_item,
                    question.history,
                    using_previous_context=False
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
                    "fuente": main_item["fuente"]
                }

            items_independientes = [ID_TO_ITEM[m] for m in matched_ids]
            consultas = build_independent_responses(question.query, items_independientes, question.history)
            texto_combinado = "\n\n".join(f"Sobre {c['tramite']}: {c['respuesta']}" for c in consultas)

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
                "consultas": consultas
            }

        expanded_ids = expand_dependencies(matched_ids)
        ordered_ids = topological_order(expanded_ids)
        ordered_items = [ID_TO_ITEM[i] for i in ordered_ids if i in ID_TO_ITEM]

        intro = generate_plan_intro(question.query, ordered_items)
        plan_text = build_plan_text(ordered_items)
        respuesta_ia = f"{intro}\n\n{plan_text}" if intro else plan_text

        return {
            "modo": "plan",
            "tramite": " + ".join(item["titulo"] for item in ordered_items),
            "respuesta": respuesta_ia,
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
            ]
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