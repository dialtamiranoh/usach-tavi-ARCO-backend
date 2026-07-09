"""
dashboard.py — Backend de benchmark ARCO, independiente de main.py
Puerto: 8090  →  uvicorn dashboard:app --port 8090 --reload

Responsabilidades (separadas de main.py):
  - Buscar archivos .gguf en el sistema (carpetas configurables).
  - Levantar / detener servidores llama.cpp (llama_cpp.server) por modelo,
    asignando puerto automáticamente.
  - Enviar la misma consulta a N modelos seleccionados y medir métricas,
    incluyendo latencia total, time-to-first-token (TTFT) y tokens/seg.
  - Persistir métricas y feedback para historial y comparación.

main.py conserva únicamente el servicio de chat en producción (/ask, /feedback
de WhatsApp) contra los modelos fijos configurados en .env. Este archivo no
importa nada de main.py: es completamente autónomo.

Archivos de persistencia:
  dashboard_metrics.jsonl  — una línea por (trace_id, alias)
  dashboard_feedback.jsonl — feedback por (trace_id, alias)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import sys
import time
import uuid
import unicodedata
import subprocess
import threading
import requests
from datetime import datetime

BASE_DIR      = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
PROMPTS_PATH   = BASE_DIR / "prompts.json"
METRICS_PATH   = BASE_DIR / "dashboard_metrics.jsonl"
FEEDBACK_PATH  = BASE_DIR / "dashboard_feedback.jsonl"
LOGS_DIR       = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    _prompts_data = json.load(f)
_version_activa = os.getenv("PROMPT_VERSION", _prompts_data["version_activa"])
SYSTEM_PROMPT = _prompts_data["versiones"][_version_activa]["system_prompt"]

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Carpetas donde se buscan .gguf por defecto. Se puede agregar más vía
# DASHBOARD_MODEL_DIRS (separadas por ";") o desde la propia UI (extra_dirs).
_default_dirs = [
    BASE_DIR.parent / "modelos",
    Path("C:/Users/diego.altamirano/models")
]
_env_dirs = [Path(p) for p in os.getenv("DASHBOARD_MODEL_DIRS", "").split(";") if p.strip()]
MODEL_SEARCH_DIRS = _default_dirs + _env_dirs

LLAMA_HOST            = os.getenv("DASHBOARD_LLAMA_HOST", "127.0.0.1")
LLAMA_PORT_START      = int(os.getenv("DASHBOARD_LLAMA_PORT_START", "8600"))
LLAMA_N_CTX           = int(os.getenv("DASHBOARD_N_CTX", "2048"))
LLAMA_HEALTH_TIMEOUT  = int(os.getenv("DASHBOARD_HEALTH_TIMEOUT", "180"))

# Hilos de CPU detectados en esta máquina. Por defecto, cada llama_cpp.server
# reserva la mitad de los cores para generación y TODOS los cores para el
# procesamiento del prompt (n_threads_batch) — si se corren 2+ modelos a la
# vez, ambos compiten por los mismos cores justo en la ventana que mide el
# TTFT, ensuciando la comparación. DASHBOARD_N_THREADS fuerza un valor fijo;
# si no se define, se reparte CPU_THREADS entre los modelos lanzados juntos.
CPU_THREADS      = os.cpu_count() or 4
FIXED_N_THREADS  = os.getenv("DASHBOARD_N_THREADS")
FIXED_N_THREADS  = int(FIXED_N_THREADS) if FIXED_N_THREADS else None

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "140"))

app = FastAPI(title="ARCO Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Registro de modelos en ejecución (en memoria)
# ---------------------------------------------------------------------------

REGISTRY: dict[str, dict] = {}
REGISTRY_LOCK = threading.Lock()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "modelo"


def unique_alias(base: str) -> str:
    alias = base
    i = 2
    while alias in REGISTRY:
        alias = f"{base}-{i}"
        i += 1
    return alias


def find_free_port() -> int:
    used = {entry["port"] for entry in REGISTRY.values()}
    port = LLAMA_PORT_START
    while port in used:
        port += 1
    return port


def poll_until_ready(alias: str) -> None:
    entry = REGISTRY.get(alias)
    if not entry:
        return
    url = f"http://{LLAMA_HOST}:{entry['port']}/v1/models"
    deadline = time.time() + LLAMA_HEALTH_TIMEOUT
    while time.time() < deadline:
        proc = entry["process"]
        if proc.poll() is not None:
            entry["status"] = "error"
            entry["error"] = f"el proceso terminó (code {proc.returncode}); ver {entry['log_path']}"
            return
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                entry["status"] = "ready"
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    entry["status"] = "error"
    entry["error"] = "timeout esperando que el modelo esté listo"


def stop_model(alias: str) -> None:
    entry = REGISTRY.get(alias)
    if not entry:
        return
    proc = entry.get("process")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    REGISTRY.pop(alias, None)


@app.on_event("shutdown")
def _stop_all_models():
    with REGISTRY_LOCK:
        for alias in list(REGISTRY.keys()):
            stop_model(alias)


# ---------------------------------------------------------------------------
# Helpers de texto (independientes de main.py)
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


def build_history_text(history: list) -> str:
    if not history:
        return "sin historial previo"
    return "\n".join(f"{m.role}: {m.content}" for m in history[-6:])


def find_item(query_norm: str) -> Optional[dict]:
    for item in KNOWLEDGE:
        for kw in item.get("keywords", []):
            if normalize_text(kw) in query_norm:
                return item
    return None


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
# Modelos Pydantic
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    extra_dirs: list[str] = Field(default_factory=list)


class LaunchRequest(BaseModel):
    paths:     list[str]
    n_ctx:     Optional[int] = None
    n_threads: Optional[int] = None  # fuerza un valor; si no, se reparte CPU_THREADS


class StopRequest(BaseModel):
    alias: str


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query:    str
    aliases:  list[str]
    history:  list[ChatMessage] = Field(default_factory=list)
    trace_id: Optional[str] = None


class Feedback(BaseModel):
    trace_id: str
    alias:    str
    score:    float
    comment:  Optional[str] = None


# ---------------------------------------------------------------------------
# Descubrimiento de modelos .gguf
# ---------------------------------------------------------------------------

def scan_gguf_files(extra_dirs: Optional[list[str]] = None) -> list[dict]:
    dirs = list(MODEL_SEARCH_DIRS)
    dirs += [Path(p) for p in (extra_dirs or []) if p.strip()]

    seen: set[Path] = set()
    results = []
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        for p in d.rglob("*.gguf"):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            results.append({
                "path":     str(rp),
                "filename": p.name,
                "dir":      str(p.parent),
                "size_mb":  round(p.stat().st_size / (1024 * 1024), 1),
                "alias":    slugify(p.stem),
            })
    results.sort(key=lambda r: r["filename"].lower())
    return results


@app.get("/gguf/scan")
def gguf_scan(extra_dirs: str = ""):
    """
    extra_dirs: rutas adicionales separadas por ';' para sumar a
    MODEL_SEARCH_DIRS en esta búsqueda puntual.
    """
    extra = [p for p in extra_dirs.split(";") if p.strip()]
    return {
        "search_dirs": [str(d) for d in MODEL_SEARCH_DIRS] + extra,
        "models": scan_gguf_files(extra),
    }


# ---------------------------------------------------------------------------
# Ciclo de vida de los modelos (levantar / detener)
# ---------------------------------------------------------------------------

def _serialize_entry(alias: str, entry: dict) -> dict:
    return {
        "alias":      alias,
        "path":       entry["path"],
        "filename":   Path(entry["path"]).name,
        "port":       entry["port"],
        "status":     entry["status"],
        "error":      entry.get("error"),
        "started_at": entry["started_at"],
        "log_path":   entry["log_path"],
        "n_threads":  entry.get("n_threads"),
    }


@app.get("/models/status")
def models_status():
    with REGISTRY_LOCK:
        return [_serialize_entry(a, e) for a, e in REGISTRY.items()]


@app.get("/system/info")
def system_info():
    with REGISTRY_LOCK:
        active = sum(1 for e in REGISTRY.values() if e["status"] in ("starting", "ready"))
    return {"cpu_threads": CPU_THREADS, "modelos_activos": active}


def _launch_one(path: Path, n_ctx: Optional[int], n_threads: int) -> str:
    """Lanza un .gguf y devuelve su alias. Debe llamarse con REGISTRY_LOCK tomado."""
    alias = unique_alias(slugify(path.stem))
    port  = find_free_port()
    log_path = LOGS_DIR / f"{alias}.log"

    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(path),
        "--host", LLAMA_HOST,
        "--port", str(port),
        "--model_alias", alias,
        "--n_ctx", str(n_ctx or LLAMA_N_CTX),
        "--n_threads", str(n_threads),
        "--n_threads_batch", str(n_threads),
    ]
    log_file = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    REGISTRY[alias] = {
        "path":       str(path),
        "port":       port,
        "process":    proc,
        "status":     "starting",
        "error":      None,
        "started_at": datetime.now().isoformat(),
        "log_path":   str(log_path),
        "n_threads":  n_threads,
    }
    return alias


@app.post("/models/launch")
def models_launch(req: LaunchRequest):
    if not req.paths:
        raise HTTPException(400, "Debes indicar al menos un archivo .gguf")

    paths = []
    for p in req.paths:
        path = Path(p)
        if path.suffix.lower() != ".gguf" or not path.is_file():
            raise HTTPException(400, f"Archivo .gguf no encontrado: {p}")
        paths.append(path)

    results: list[dict] = []
    new_aliases: list[str] = []

    with REGISTRY_LOCK:
        already_running = 0
        to_launch = []
        for path in paths:
            existing = next(
                ((a, e) for a, e in REGISTRY.items()
                 if Path(e["path"]) == path and e["status"] in ("starting", "ready")),
                None,
            )
            if existing:
                results.append(_serialize_entry(*existing))
                already_running += 1
            else:
                to_launch.append(path)

        # Además de los modelos ya corriendo fuera de este lote, cuenta los
        # que ya estaban activos: todos compiten por los mismos cores.
        other_running = sum(
            1 for e in REGISTRY.values()
            if e["status"] in ("starting", "ready") and Path(e["path"]) not in paths
        )
        total_concurrent = other_running + already_running + len(to_launch)

        if req.n_threads:
            n_threads = req.n_threads
        elif FIXED_N_THREADS:
            n_threads = FIXED_N_THREADS
        else:
            n_threads = max(1, CPU_THREADS // max(total_concurrent, 1))

        for path in to_launch:
            alias = _launch_one(path, req.n_ctx, n_threads)
            new_aliases.append(alias)
            results.append(_serialize_entry(alias, REGISTRY[alias]))

    for alias in new_aliases:
        threading.Thread(target=poll_until_ready, args=(alias,), daemon=True).start()

    return results


@app.post("/models/stop")
def models_stop(req: StopRequest):
    with REGISTRY_LOCK:
        if req.alias not in REGISTRY:
            raise HTTPException(404, f"Modelo no registrado: {req.alias}")
        stop_model(req.alias)
    return {"status": "stopped", "alias": req.alias}


# ---------------------------------------------------------------------------
# Llamada al LLM con streaming (para medir TTFT)
# ---------------------------------------------------------------------------

def call_llm_stream(alias: str, entry: dict, user_query: str, item: Optional[dict], history: list) -> dict:
    url = f"http://{LLAMA_HOST}:{entry['port']}/v1/chat/completions"

    history_text = build_history_text(history)
    if item:
        context_text = f"""
            tramite: {item['titulo']}
            respuesta base: {item['respuesta']}
            costo: {item.get('costo', 'no especificado')}
            duracion: {item.get('duracion', 'no especificada')}
            canal: {item.get('canal', 'no especificado')}
            presencialidad: {item.get('presencialidad', 'no especificada')}
            requiere clave unica: {item.get('requiere_clave_unica', 'no especificado')}
            fuente oficial: {item.get('fuente', 'no especificada')}
        """.strip()
    else:
        context_text = "no se identifico un tramite especifico para esta consulta."

    user_prompt = f"""
pregunta actual del usuario: {user_query}

historial reciente:
{history_text}

contexto del tramite:
{context_text}

redacta una respuesta breve de orientacion para el usuario.
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    payload = {
        "model":       alias,
        "messages":    messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens":  LLM_MAX_TOKENS,
        "stream":      True,
    }

    input_tokens_est = estimate_tokens(json.dumps(messages, ensure_ascii=False))
    start = time.time()
    ttft_ms = None
    chunks: list[str] = []
    error = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or [{}]
                piece = choices[0].get("delta", {}).get("content")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.time() - start) * 1000
                    chunks.append(piece)
        respuesta = clean_model_text("".join(chunks).strip())
    except Exception as e:
        respuesta = None
        error     = str(e)

    total_ms = (time.time() - start) * 1000
    # llama_cpp.server no reporta "usage" en modo streaming: se estima por caracteres.
    output_tokens = estimate_tokens(respuesta) if respuesta else 0
    input_tokens  = input_tokens_est
    total_tokens  = input_tokens + output_tokens

    gen_ms = max(total_ms - (ttft_ms or 0), 0)
    tokens_per_sec = round(output_tokens / (gen_ms / 1000), 2) if gen_ms > 0 and output_tokens else 0

    return {
        "respuesta":       respuesta,
        "latency_ms":      round(total_ms, 1),
        "ttft_ms":         round(ttft_ms, 1) if ttft_ms is not None else None,
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "total_tokens":    total_tokens,
        "tokens_per_sec":  tokens_per_sec,
        "error":           error,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    with REGISTRY_LOCK:
        modelos_activos = [_serialize_entry(a, e) for a, e in REGISTRY.items()]
    return {"message": "ARCO dashboard backend", "modelos_activos": modelos_activos}


@app.post("/ask")
def ask(req: AskRequest):
    if not req.aliases:
        raise HTTPException(400, "Debes seleccionar al menos un modelo")

    trace_id   = req.trace_id or str(uuid.uuid4())
    query_norm = normalize_text(req.query)
    item       = find_item(query_norm)
    timestamp  = datetime.now().isoformat()

    with REGISTRY_LOCK:
        targets = {}
        missing = []
        for alias in req.aliases:
            entry = REGISTRY.get(alias)
            if not entry or entry["status"] != "ready":
                missing.append(alias)
            else:
                targets[alias] = dict(entry)

    results = {}
    for alias in missing:
        results[alias] = {"error": "modelo no está listo (ver /models/status)"}

    if targets:
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            futures = {
                alias: pool.submit(call_llm_stream, alias, entry, req.query, item, req.history)
                for alias, entry in targets.items()
            }
            for alias, future in futures.items():
                results[alias] = future.result()

    for alias, r in results.items():
        if "error" in r and "respuesta" not in r:
            continue
        save_metric({
            "trace_id":      trace_id,
            "timestamp":     timestamp,
            "alias":         alias,
            "model_path":    targets.get(alias, {}).get("path"),
            "tramite":       item["titulo"] if item else "No identificado",
            "user_query":    req.query,
            **r,
        })

    return {
        "trace_id": trace_id,
        "tramite":  item["titulo"] if item else "No identificado",
        "results":  results,
    }


@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    save_feedback_entry({
        "trace_id":  feedback.trace_id,
        "alias":     feedback.alias,
        "score":     feedback.score,
        "comment":   feedback.comment,
        "timestamp": datetime.now().isoformat(),
    })
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Métricas y comparación
# ---------------------------------------------------------------------------

@app.get("/metrics")
def get_metrics():
    metrics   = load_metrics()
    feedbacks = load_feedbacks()
    fb_index  = {(fb["trace_id"], fb["alias"]): fb for fb in feedbacks}
    for m in metrics:
        m["feedback"] = fb_index.get((m["trace_id"], m["alias"]))
    return metrics


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0
    idx = max(int(len(sorted_vals) * pct) - 1, 0)
    return sorted_vals[idx]


@app.get("/stats/compare")
def get_stats_compare():
    metrics   = load_metrics()
    feedbacks = load_feedbacks()
    fb_index  = {(fb["trace_id"], fb["alias"]): fb for fb in feedbacks}

    aliases = sorted({m["alias"] for m in metrics})
    result = {}
    for alias in aliases:
        subset = [m for m in metrics if m["alias"] == alias]

        latencies = sorted(m["latency_ms"] for m in subset if m.get("latency_ms"))
        ttfts     = sorted(m["ttft_ms"] for m in subset if m.get("ttft_ms") is not None)
        tps_vals  = [m["tokens_per_sec"] for m in subset if m.get("tokens_per_sec")]

        total_tok = sum(m.get("total_tokens", 0) for m in subset)

        fb_subset = [fb_index[(m["trace_id"], m["alias"])] for m in subset if (m["trace_id"], m["alias"]) in fb_index]
        positive  = sum(1 for fb in fb_subset if fb["score"] >= 0.8)
        negative  = sum(1 for fb in fb_subset if fb["score"] <= 0.2)
        neutral   = len(fb_subset) - positive - negative

        result[alias] = {
            "total_consultas":       len(subset),
            "promedio_latencia_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p95_latencia_ms":       round(_percentile(latencies, 0.95), 1),
            "promedio_ttft_ms":      round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
            "p95_ttft_ms":           round(_percentile(ttfts, 0.95), 1) if ttfts else None,
            "promedio_tokens_seg":   round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else 0,
            "total_tokens":          total_tok,
            "tokens_por_consulta":   round(total_tok / len(subset), 1) if subset else 0,
            "feedback_positivos":    positive,
            "feedback_negativos":    negative,
            "feedback_neutral":      neutral,
            "pct_feedback_positivo": round(positive / len(fb_subset) * 100, 1) if fb_subset else 0,
        }

    return result


@app.get("/metrics/timeline")
def get_timeline():
    metrics = load_metrics()
    aliases = sorted({m["alias"] for m in metrics})
    result  = {}
    for alias in aliases:
        subset = [m for m in metrics if m["alias"] == alias]
        subset.sort(key=lambda x: x.get("timestamp", ""))
        result[alias] = [
            {
                "timestamp":      m["timestamp"],
                "latency_ms":     round(m.get("latency_ms", 0), 1),
                "ttft_ms":        m.get("ttft_ms"),
                "tokens_per_sec": m.get("tokens_per_sec", 0),
                "tokens":         m.get("total_tokens", 0),
                "tramite":        m.get("tramite", ""),
            }
            for m in subset[-50:]
        ]
    return result
