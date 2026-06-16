import pytest
from main import call_llm

INSTRUCCIONES_MINIMAS = [
    "no inventes",
    "fuente oficial",
    "orientacion",
    "registro civil",
    "espanol",
]

def get_system_prompt():
    """Extrae el system_prompt desde la función call_llm en main.py"""
    import inspect
    source = inspect.getsource(call_llm)
    start = source.find('system_prompt = (') 
    end = source.find('history_text =')
    return source[start:end]

def test_system_prompt_contiene_instrucciones_minimas():
    prompt = get_system_prompt().lower()
    faltantes = []
    for instruccion in INSTRUCCIONES_MINIMAS:
        if instruccion not in prompt:
            faltantes.append(instruccion)
    assert not faltantes, f"El system_prompt falta instrucciones: {faltantes}"

def test_system_prompt_no_esta_vacio():
    prompt = get_system_prompt()
    assert len(prompt) > 100, "El system_prompt es demasiado corto"

def test_system_prompt_limita_respuesta():
    prompt = get_system_prompt().lower()
    assert any(p in prompt for p in ["maximo", "máximo", "3 oraciones", "parrafo"]), \
        "El system_prompt debe limitar la longitud de la respuesta"