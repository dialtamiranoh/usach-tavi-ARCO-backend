# ARCO — Asistente para el Registro Civil y su Orientación

**Rama:** `feature/dynamic-llm`  
**Sprint:** 2  
**Última actualización:** Junio 2026

---

## ¿Qué es ARCO?

ARCO es un asistente conversacional que responde preguntas en lenguaje natural sobre trámites del Servicio de Registro Civil e Identificación de Chile. Su propósito es social: reducir fricción, evitar desplazamientos innecesarios y orientar a ciudadanos sobre requisitos, costos, canales y tiempos de entrega de cada trámite.

**ARCO no ejecuta trámites ni captura datos personales. Es un sistema de orientación.**

---

## Novedades de esta versión (Sprint 2)

### 1. Configuración dinámica del LLM via variables de entorno
El modelo de lenguaje ya no está hardcodeado en el código. Se configura mediante un archivo `.env`, lo que permite cambiar entre modelos (Qwen, Granite, u otros) sin tocar `main.py`.

### 2. RAG con ChromaDB e ingestión dinámica
Se incorporó un pipeline de Retrieval-Augmented Generation (RAG) usando ChromaDB como base de datos vectorial y `paraphrase-multilingual-MiniLM-L12-v2` como modelo de embeddings. Esto permite búsqueda semántica sobre el corpus oficial.

### 3. Script de ingestión `ingest.py`
Permite agregar nuevos documentos (PDF, TXT) al corpus sin tocar el código. Incluye control de archivos ya procesados mediante hash MD5 para evitar reingestiones innecesarias.

### 4. Keywords enriquecidas en `knowledge.json`
Todos los trámites fueron enriquecidos con keywords semánticas adicionales para mejorar la precisión de la búsqueda RAG. Por ejemplo, "pasaporte" ahora incluye "viajar fuera de chile", "salir del pais", "documento para viajar".

### 5. Prompt con clarificación de ambigüedades
El sistema prompt fue actualizado para que ARCO haga preguntas de clarificación cuando la consulta es ambigua (por ejemplo, cuando no queda claro si el usuario es chileno o extranjero).

### 6. CI/CD con GitHub Actions
Pipeline de integración continua que se ejecuta automáticamente en cada push. Valida:
- Sintaxis de `main.py`
- Estructura y campos obligatorios de `knowledge.json`
- Instalación de dependencias

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Backend | Python 3.11 + FastAPI + Uvicorn |
| Frontend | HTML5 + CSS3 + JavaScript vanilla |
| LLM local | llama-cpp-python server (puerto 8001) |
| Modelo por defecto | Qwen2.5-3B-Instruct Q4_K_M GGUF |
| RAG | ChromaDB + sentence-transformers |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Recuperación (fallback) | Keywords normalizados sobre knowledge.json |
| CI/CD | GitHub Actions |
| Control de versiones | GitHub (2 repositorios: frontend + backend) |

---

## Arquitectura

![Arquitectura ARCO](Arquitectura.png)

---

## Requisitos previos

- Python 3.11+
- Git
- 16 GB RAM recomendados
- 10 GB de espacio libre en disco
- Modelo `.gguf` descargado localmente
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
# Crear venv en la carpeta padre
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar dependencias del backend

```bash
cd usach-tavi-ARCO-backend
pip install -r requirements.txt
pip install "llama-cpp-python[server]"
```

### 4. Configurar variables de entorno

```bash
# Copiar el archivo de ejemplo
cp .env.example .env
```

Editar `.env` con la ruta real al modelo:

```env
LLM_URL=http://127.0.0.1:8001/v1/chat/completions
LLM_MODEL=arco-llm
LLM_MODEL_PATH=C:/Users/TU_USUARIO/models/qwen2.5-3b-instruct-q4_k_m.gguf
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=140
LLM_N_CTX=2048
USE_RAG=false
```

### 5. Descargar el modelo

```bash
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir "C:/Users/TU_USUARIO/models"
```

### 6. Ingestar el corpus (solo si USE_RAG=true)

```bash
python ingest.py
```

---

## Ejecución

Abrir tres terminales con el venv activo:

### Terminal 1 — Modelo LLM

```bash
python -m llama_cpp.server \
  --model "RUTA/AL/MODELO.gguf" \
  --host 127.0.0.1 \
  --port 8001 \
  --model_alias arco-llm \
  --n_ctx 2048
```

### Terminal 2 — Backend

```bash
cd usach-tavi-ARCO-backend
uvicorn main:app --reload
```

### Terminal 3 — Frontend

```bash
cd usach-tavi-ARCO-frontend
python -m http.server 5500
```

Abrir en el navegador: `http://127.0.0.1:5500`

---

## Uso del sistema

### Consultas de ejemplo

| Consulta | Respuesta esperada |
|---|---|
| "quiero sacar mi pasaporte" | Canal mixto, costo, plazo 8 días hábiles, fuente oficial |
| "¿cuánto cuesta el certificado de antecedentes?" | Gratis en línea / $1.050 en oficina |
| "soy extranjero y necesito sacar mi cédula" | Reserva de hora, comparecencia presencial, 20 días hábiles |
| "necesito un papel para viajar fuera de Chile" | Pasaporte (con RAG activado) |
| "quiero renovar mi licencia de conducir" | Fallback: fuera del dominio de ARCO |

### Cambiar de modelo

Para usar Granite u otro modelo compatible:

1. Descargar el modelo `.gguf`
2. Editar `.env`:
```env
LLM_MODEL_PATH=C:/Users/TU_USUARIO/models/granite-3.1-3b-a800m-instruct-q4_k_m.gguf
```
3. Reiniciar el servidor del modelo (Terminal 1)

### Agregar documentos al corpus RAG

1. Copiar archivos `.pdf` o `.txt` a la carpeta `docs/`
2. Ejecutar:
```bash
python ingest.py
```
3. Reiniciar el backend

El script detecta automáticamente si un archivo ya fue ingestado y lo omite si no cambió.

---

## API

### GET /
Verifica que el backend está funcionando.

### POST /ask
Recibe una consulta y devuelve orientación sobre el trámite.

**Request:**
```json
{
  "query": "quiero sacar mi pasaporte",
  "history": [],
  "context": null
}
```

**Response:**
```json
{
  "tramite": "Pasaporte",
  "respuesta": "Para obtener o renovar tu pasaporte...",
  "respuesta_base": "El pasaporte es el documento de viaje...",
  "costo": "$69.660 (32 paginas) / $69.740 (64 paginas)",
  "duracion": "El plazo de entrega es de 8 dias habiles...",
  "canal": "mixto",
  "presencialidad": "si",
  "requiere_clave_unica": "si",
  "fuente": "https://www.chileatiende.gob.cl/fichas/3445-pasaporte"
}
```

---

## Mejoras pendientes

### Alta prioridad (Sprint 2)
- [ ] Lógica de detección de ambigüedad en `main.py` con campo `ambiguo` en `knowledge.json`
- [ ] Benchmark comparativo Qwen2.5-3B vs Granite-3.1-3B (tiempo de respuesta y calidad)
- [ ] Soporte para archivos `.docx` en el ingestor
- [ ] Merge de `feature/dynamic-llm` a `main` con pruebas de regresión

### Media prioridad (Sprint 3)
- [ ] Scraping directo de ChileAtiende para actualización automática del corpus
- [ ] Historial persistente de conversaciones (SQLite)
- [ ] Panel de administración para gestionar el corpus sin tocar archivos
- [ ] Despliegue web en Railway o Render con GPT-4o mini como LLM

### Baja prioridad (post-curso)
- [ ] Integración con ClaveÚnica para trámites personalizados
- [ ] Soporte multiidioma (mapudungun, inglés para turistas)
- [ ] Base de datos vectorial en producción con actualización periódica automática

---

## Estructura del repositorio

```
usach-tavi-ARCO-backend/
├── main.py              # Backend FastAPI con lógica RAG y keyword search
├── ingest.py            # Script de ingestión dinámica de documentos
├── knowledge.json       # Corpus de 21 trámites del Registro Civil
├── requirements.txt     # Dependencias Python
├── .env.example         # Plantilla de variables de entorno
├── .gitignore           # Archivos excluidos del repositorio
├── docs/                # Carpeta para documentos adicionales (no se sube)
├── chroma_db/           # Base de datos vectorial ChromaDB (no se sube)
└── .github/
    └── workflows/
        └── ci.yml       # Pipeline CI/CD GitHub Actions
```

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






# ARCO — Guía de ejecución con benchmark

## Arquitectura

```
Granite (llama-server :8001) ──┐
                                ├── main.py (:8000) ── index.html
Qwen    (llama-server :8002) ──┘                  └── dashboard.html
```

Cada consulta enviada desde `index.html` llama a **ambos modelos en paralelo**.  
La respuesta mostrada corresponde al modelo seleccionado; la del otro se guarda silenciosamente para benchmark.

---

## Requisitos previos

- Python 3.11+ con entorno virtual activado
- `llama-server` (llama.cpp) disponible en el PATH
- Modelos descargados en `../modelos/`

---

## Paso 1 — Servidor Granite (terminal 1)

```bash
llama-server \
  --model "../modelos/granite/granite-3.3-2b-instruct-Q4_K_M.gguf" \
  --port 8001 \
  --ctx-size 2048 \
  --n-predict 140 \
  -ngl 0
```

Verificar: http://127.0.0.1:8001/health → `{"status":"ok"}`

---

## Paso 2 — Servidor Qwen (terminal 2)

```bash
llama-server \
  --model "../modelos/qwen/Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
  --port 8002 \
  --ctx-size 2048 \
  --n-predict 140 \
  -ngl 0
```

Verificar: http://127.0.0.1:8002/health → `{"status":"ok"}`

---

## Paso 3 — Backend ARCO (terminal 3)

```bash
cd usach-tavi-ARCO-backend
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

uvicorn main:app --port 8000 --reload
```

Verificar: http://127.0.0.1:8000 → lista los dos modelos activos

---

## Paso 4 — Frontend

Abrir con Live Server (VS Code) o directamente en el navegador:

| Archivo | URL | Función |
|---|---|---|
| `index.html` | http://127.0.0.1:5500/index.html | Chat con selector de modelo |
| `dashboard.html` | http://127.0.0.1:5500/dashboard.html | Dashboard de benchmark |

---

## Variables de entorno (`.env`)

```env
GRANITE_URL=http://127.0.0.1:8001/v1/chat/completions
GRANITE_MODEL=granite-3.1-3b-instruct

QWEN_URL=http://127.0.0.1:8002/v1/chat/completions
QWEN_MODEL=qwen2.5-3b-instruct

LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=140

USE_RAG=false
```

---

## Archivos de datos

| Archivo | Contenido |
|---|---|
| `benchmark_metrics.jsonl` | Una línea por consulta con latencia, tokens, modelo y trámite |
| `benchmark_feedback.jsonl` | Feedback (👍/🤔/👎) por `trace_id` |

---

## Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/ask` | Consulta al modelo (`model: "granite"` o `"qwen"`) |
| `GET` | `/models` | Lista modelos configurados |
| `POST` | `/feedback` | Registra feedback de una respuesta |
| `GET` | `/metrics` | Todas las métricas con feedback |
| `GET` | `/stats` | Resumen global |
| `GET` | `/stats/compare` | Comparación por modelo (dashboard) |
| `GET` | `/metrics/timeline` | Latencia en el tiempo por modelo |
