import json
import pytest
from pathlib import Path

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.json"

INSTRUCCIONES_MINIMAS = [
    "no inventes",
    "fuente oficial",
    "orientacion",
    "registro civil",
    "espanol",
]

@pytest.fixture
def prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_system_prompt(prompts):
    version_activa = prompts["version_activa"]
    return prompts["versiones"][version_activa]["system_prompt"]

def test_system_prompt_contiene_instrucciones_minimas(prompts):
    prompt = get_system_prompt(prompts).lower()
    faltantes = [i for i in INSTRUCCIONES_MINIMAS if i not in prompt]
    assert not faltantes, f"El system_prompt falta instrucciones: {faltantes}"

def test_system_prompt_no_esta_vacio(prompts):
    prompt = get_system_prompt(prompts)
    assert len(prompt) > 100, "El system_prompt es demasiado corto"

def test_system_prompt_limita_respuesta(prompts):
    prompt = get_system_prompt(prompts).lower()
    assert any(p in prompt for p in ["maximo", "máximo", "3 oraciones", "parrafo"]), \
        "El system_prompt debe limitar la longitud de la respuesta"