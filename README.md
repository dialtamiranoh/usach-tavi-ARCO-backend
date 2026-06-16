# ARCO — Asistente para el Registro Civil y su Orientación

**Rama activa:** `feature/mlops-pipeline`  
**Sprint:** 2  
**Última actualización:** Junio 2026

---

## ¿Qué es ARCO?

ARCO es un asistente conversacional que responde preguntas en lenguaje natural sobre trámites del Servicio de Registro Civil e Identificación de Chile. Su propósito es social: reducir fricción, evitar desplazamientos innecesarios y orientar a ciudadanos sobre requisitos, costos, canales y tiempos de entrega de cada trámite.

**ARCO no ejecuta trámites ni captura datos personales. Es un sistema de orientación.**

---

## Novedades Sprint 2

### 1. Soporte dual de modelos LLM (Granite y Qwen)
El backend soporta dos modelos simultáneamente configurables via `.env`. Cada consulta registra el modelo utilizado en el benchmark. El frontend permite seleccionar el modelo activo y ejecuta silenciosamente el otro para comparación.

### 2. Benchmark automático con dashboard
Cada consulta se registra en `benchmark_metrics.jsonl` con latencia, tokens, modelo y trámite. El feedback del usuario (👍/🤔/👎) se guarda en `benchmark_feedback.jsonl`. El dashboard `dashboard.html` muestra comparación en tiempo real entre Granite y Qwen.

### 3. RAG con ChromaDB e ingestión dinámica
Pipeline de Retrieval-Augmented Generation usando ChromaDB y embeddings `paraphrase-multilingual-MiniLM-L12-v2`. El script `ingest.py` procesa `knowledge.json` y documentos adicionales (PDF, TXT) desde la carpeta `docs/` con deduplicación por hash MD5.

### 4. Versionado de prompts
El system prompt se gestiona en `prompts.json` con soporte multi-versión. La versión activa se controla via `PROMPT_VERSION` en `.env` sin tocar el código. Versiones disponibles: v1.0.0 (base) y v1.1.0 (con clarificación de nacionalidad).

### 5. Pipeline MLOps con tests automáticos
16 tests pytest organizados en tres módulos — conocimiento, prompts y smoke tests — que validan el sistema sin necesitar el LLM corriendo. El pipeline CI/CD corre automáticamente en cada push.

### 6. Evaluación automática con ground truth
10 casos de prueba con trámite esperado y keywords de respuesta. El evaluador genera reportes con timestamp y mantiene un historial. Accuracy actual: 80% (baseline documentado).

### 7. Detección de regresión
Comparación automática contra un baseline aprobado. Si un caso que antes funcionaba deja de funcionar, el pipeline falla y bloquea el merge.

### 8. Observabilidad con logging estructurado
Logging en `arco.log` con formato `timestamp INFO query=... modelo=... tramite=... rag=... followup=...`. Librerías externas (HuggingFace, ChromaDB, httpx) filtradas para mantener el log limpio.

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Backend | Python 3.11 + FastAPI + Uvicorn |
| Frontend | HTML5 + CSS3 + JavaScript vanilla |
| LLM — Granite | llama-cpp-python server :8001 |
| LLM — Qwen | llama-cpp-python server :8002 |
| Modelos | Granite-3.1-3B Q4_K_M / Qwen2.5-3B Q4_K_M GGUF |
| RAG | ChromaDB + sentence-transformers |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Recuperación fallback | Keywords normalizados sobre knowledge.json |
| CI/CD | GitHub Actions |
| Testing | pytest + httpx + FastAPI TestClient |
| Observabilidad | logging estructurado en arco.log |

---

## Arquitectura

![Arquitectura ARCO](Arquitectura.png)

---

## Requisitos previos

- Python 3.11+
- Git
- 16 GB RAM recomendados
- 10 GB de espacio libre en disco
- Modelos `.gguf` descargados localmente
- Windows 10+ / Linux / macOS

---

## Instalación

### 1. Clonar los repositorios

```bash
git clone https://github.com/dialtamiranoh/usach-tavi-ARCO-backend.git
git clone https://github.com/dialtamiranoh/usach-tavi-ARCO-frontend.git
```

### 2. Crear y activar el entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
cd usach-tavi-ARCO-backend
pip install -r requirements.txt
pip install "llama-cpp-python[server]"
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` con las rutas reales a los modelos:

```env
GRANITE_URL=http://127.0.0.1:8001/v1/chat/completions
GRANITE_MODEL=granite-3.1-3b-instruct
QWEN_URL=http://127.0.0.1:8002/v1/chat/completions
QWEN_MODEL=qwen2.5-3b-instruct
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=140
USE_RAG=false
PROMPT_VERSION=1.1.0
```

### 5. Ingestar el corpus (solo si USE_RAG=true)

```bash
python ingest.py
```

---

## Ejecución

Abrir cuatro terminales con el venv activo:

### Terminal 1 — Modelo Granite

```bash
python -m llama_cpp.server \
  --model "RUTA/granite-3.1-3b-a800m-instruct-q4_k_m.gguf" \
  --host 127.0.0.1 --port 8001 \
  --model_alias granite-3.1-3b-instruct --n_ctx 2048
```

Verificar: `http://127.0.0.1:8001/health`

### Terminal 2 — Modelo Qwen

```bash
python -m llama_cpp.server \
  --model "RUTA/qwen2.5-3b-instruct-q4_k_m.gguf" \
  --host 127.0.0.1 --port 8002 \
  --model_alias qwen2.5-3b-instruct --n_ctx 2048
```

Verificar: `http://127.0.0.1:8002/health`

### Terminal 3 — Backend

```bash
cd usach-tavi-ARCO-backend
uvicorn main:app --port 8000 --reload
```

Verificar: `http://127.0.0.1:8000`

### Terminal 4 — Frontend

```bash
cd usach-tavi-ARCO-frontend
python -m http.server 5500
```

Abrir en el navegador:
- Chat: `http://127.0.0.1:5500/index.html`
- Dashboard benchmark: `http://127.0.0.1:5500/dashboard.html`

---

## Uso del sistema

### Consultas de ejemplo

| Consulta | Respuesta esperada |
|---|---|
| "quiero sacar mi pasaporte" | Canal mixto, costo, plazo 8 días hábiles, fuente oficial |
| "¿cuánto cuesta el certificado de antecedentes?" | Gratis en línea / $1.050 en oficina |
| "soy extranjero y necesito sacar mi cédula" | Reserva de hora, presencial, 20 días hábiles |
| "necesito un papel para viajar fuera de Chile" | Pasaporte (con RAG activado) |
| "quiero renovar mi licencia de conducir" | Fallback: fuera del dominio de ARCO |

### Cambiar modelo activo

En `index.html` seleccionar Granite o Qwen desde el selector. El modelo no seleccionado corre en segundo plano para el benchmark.

### Cambiar versión del prompt

Editar `.env`:
```env
PROMPT_VERSION=1.0.0   # Prompt base
PROMPT_VERSION=1.1.0   # Con clarificación de nacionalidad
```

Reiniciar el backend para aplicar el cambio.

### Agregar documentos al corpus RAG

```bash
# Copiar documentos a docs/
cp manual_registro_civil.pdf docs/

# Reingestar
python ingest.py

# Reiniciar el backend
uvicorn main:app --port 8000 --reload
```

---

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Estado del backend y modelos activos |
| GET | `/models` | Lista modelos configurados |
| POST | `/ask` | Consulta al modelo seleccionado |
| POST | `/feedback` | Registra feedback de una respuesta |
| GET | `/metrics` | Todas las métricas con feedback |
| GET | `/stats` | Resumen global |
| GET | `/stats/compare` | Comparación Granite vs Qwen |
| GET | `/metrics/timeline` | Latencia en el tiempo por modelo |

### Ejemplo POST /ask

```json
{
  "query": "quiero sacar mi pasaporte",
  "model": "granite",
  "history": [],
  "context": null
}
```

```json
{
  "tramite": "Pasaporte",
  "respuesta": "Para obtener o renovar tu pasaporte...",
  "costo": "$69.660 (32 paginas) / $69.740 (64 paginas)",
  "duracion": "8 dias habiles desde la solicitud",
  "canal": "mixto",
  "presencialidad": "si",
  "requiere_clave_unica": "si",
  "fuente": "https://www.chileatiende.gob.cl/fichas/3445-pasaporte",
  "trace_id": "uuid",
  "model_used": "Granite-3.1-3B",
  "latency_ms": 1200,
  "total_tokens": 320,
  "fallback": false
}
```

---

## Tests y evaluación

### Correr tests automáticos

```bash
pytest tests/ -v
```

16 tests en 3 módulos:
- `tests/test_knowledge.py` — 6 tests (estructura del corpus)
- `tests/test_prompts.py` — 3 tests (instrucciones del system prompt)
- `tests/test_smoke.py` — 7 tests (endpoints sin LLM)

### Correr evaluador con ground truth

```bash
python eval/evaluador.py
```

Genera `eval/reporte_YYYYMMDD_HHMMSS.json` y actualiza `eval/reporte_latest.json`.

### Verificar regresiones vs baseline

```bash
python eval/verificar_baseline.py
```

Falla con exit code 1 si algún caso que antes pasaba ahora falla.

### Estado actual del evaluador

| Caso | Query | Estado |
|---|---|---|
| TC-01 | quiero sacar pasaporte | ✅ PASS |
| TC-02 | cuanto cuesta el certificado de antecedentes | ✅ PASS |
| TC-03 | renovar carnet de identidad | ✅ PASS |
| TC-04 | cedula para extranjeros | ❌ FAIL (limitación keyword search) |
| TC-05 | sacar clave unica | ✅ PASS |
| TC-06 | transferir un vehiculo | ✅ PASS |
| TC-07 | inscribir nacimiento recien nacido | ❌ FAIL (limitación keyword search) |
| TC-08 | sacar certificado de nacimiento | ✅ PASS |
| TC-09 | casarse en el registro civil | ✅ PASS |
| TC-10 | quiero renovar licencia de conducir | ✅ PASS |

TC-04 y TC-07 fallan porque el keyword search por substring no resuelve ambigüedad semántica. Se resuelven activando RAG (`USE_RAG=true`).

---

## Estructura del repositorio

```
usach-tavi-ARCO-backend/
├── main.py                  # Backend FastAPI — dual modelo, RAG, benchmark, métricas
├── ingest.py                # Ingestión dinámica PDF/TXT con hash MD5
├── knowledge.json           # Corpus 21 trámites Registro Civil
├── prompts.json             # Versionado de prompts (v1.0.0 y v1.1.0)
├── requirements.txt
├── .env.example             # Plantilla de variables de entorno
├── .gitignore
├── Arquitectura.png
├── README.md
├── tests/
│   ├── conftest.py          # Fuerza USE_RAG=false para todos los tests
│   ├── test_knowledge.py    # 6 tests validación corpus
│   ├── test_prompts.py      # 3 tests validación prompt
│   └── test_smoke.py        # 7 tests endpoint sin LLM
├── eval/
│   ├── casos_prueba.json    # 10 casos ground truth
│   ├── evaluador.py         # Evaluador automático con historial timestamp
│   ├── verificar_baseline.py # Detección de regresión vs baseline
│   ├── baseline.json        # Snapshot aprobado (no se modifica manualmente)
│   └── reporte_latest.json  # Último reporte generado
├── docs/                    # Documentos adicionales para RAG (no sube a git)
├── chroma_db/               # Base de datos vectorial ChromaDB (no sube a git)
└── .github/
    └── workflows/
        └── ci.yml           # Pipeline CI/CD GitHub Actions
```

---

## CI/CD Pipeline

El pipeline corre automáticamente en cada push a `main` y `feature/*`:

```
git push
    ↓
GitHub Actions
    ↓
  1. Instalar Python 3.11 + dependencias
  2. Validar knowledge.json (JSON válido + campos obligatorios)
  3. Validar sintaxis main.py
  4. pytest tests/ -v --tb=short (16 tests)
  5. python eval/evaluador.py (accuracy >= 90% requerida)
  6. python eval/verificar_baseline.py (sin regresiones)
```

---

## Archivos de datos (no suben a git)

| Archivo | Contenido |
|---|---|
| `benchmark_metrics.jsonl` | Una línea por consulta con latencia, tokens, modelo y trámite |
| `benchmark_feedback.jsonl` | Feedback (👍/🤔/👎) por `trace_id` |
| `arco.log` | Log estructurado de consultas |
| `chroma_db/` | Base vectorial ChromaDB |
| `docs/` | Documentos adicionales para RAG |

---

## Mejoras pendientes

### Alta prioridad
- [ ] Resolver TC-04 y TC-07 con RAG semántico activado
- [ ] Merge de `feature/mlops-pipeline` a `main`
- [ ] Benchmark formal Granite vs Qwen con 20+ consultas documentadas

### Media prioridad
- [ ] Integración WhatsApp via Twilio + n8n
- [ ] Despliegue en Railway con GPT-4o mini como LLM
- [ ] Soporte `.docx` en el ingestor

### Baja prioridad
- [ ] Historial persistente de conversaciones (SQLite)
- [ ] Scraping automático de ChileAtiende para actualización del corpus

---

## Equipo

| Integrante | Rol |
|---|---|
| Diego Altamirano Hernández | Product Owner / Analista |
| Roberto Orellana Tamayo | Backend Developer |
| Pablo Purches Zapata | IA / Datos |
| Gianello Valenzuela Robin | Frontend / QA |

**Profesor:** Daniel Gacitúa Vásquez  
**Curso:** TAVI 2026-1 — Ingeniería de Ejecución en Computación e Informática, USACH