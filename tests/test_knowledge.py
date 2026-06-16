import json
import pytest
from pathlib import Path

KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "knowledge.json"
CAMPOS_OBLIGATORIOS = [
    "id", "titulo", "keywords", "respuesta",
    "costo", "duracion", "canal",
    "presencialidad", "requiere_clave_unica", "fuente"
]

@pytest.fixture
def knowledge():
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_knowledge_es_json_valido(knowledge):
    assert isinstance(knowledge, list), "knowledge.json debe ser una lista"
    assert len(knowledge) > 0, "knowledge.json no puede estar vacío"
    print(f"\n✅ {len(knowledge)} trámites encontrados")

def test_todos_los_campos_obligatorios(knowledge):
    errores = []
    for item in knowledge:
        for campo in CAMPOS_OBLIGATORIOS:
            if campo not in item:
                errores.append(f"Trámite '{item.get('id', '?')}' falta campo: {campo}")
    assert not errores, "\n".join(errores)

def test_keywords_no_vacias(knowledge):
    errores = []
    for item in knowledge:
        if not item.get("keywords"):
            errores.append(f"Trámite '{item['id']}' tiene keywords vacías")
        elif len(item["keywords"]) < 3:
            errores.append(f"Trámite '{item['id']}' tiene menos de 3 keywords")
    assert not errores, "\n".join(errores)

def test_urls_fuente_validas(knowledge):
    errores = []
    for item in knowledge:
        fuente = item.get("fuente", "")
        if not fuente.startswith("https://"):
            errores.append(f"Trámite '{item['id']}' tiene URL inválida: {fuente}")
    assert not errores, "\n".join(errores)

def test_ids_unicos(knowledge):
    ids = [item["id"] for item in knowledge]
    duplicados = [id for id in ids if ids.count(id) > 1]
    assert not duplicados, f"IDs duplicados encontrados: {set(duplicados)}"

def test_cantidad_tramites(knowledge):
    assert len(knowledge) >= 21, f"Se esperaban al menos 21 trámites, hay {len(knowledge)}"