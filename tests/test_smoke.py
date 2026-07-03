import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

# Forzar USE_RAG=false y omitir la carga del modelo LLM para todos los tests
os.environ["USE_RAG"] = "false"
os.environ["ARCO_SKIP_MODEL_LOAD"] = "true"

from main import app

client = TestClient(app)

def test_root_responde():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "modelo" in data

def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "label" in data

def test_ask_pasaporte_keyword():
    with patch("main.call_llm") as mock_llm:
        mock_llm.return_value = ("Respuesta mock del pasaporte", {
            "latency_ms": 100, "ttft_ms": 40, "input_tokens": 10,
            "output_tokens": 20, "total_tokens": 30,
            "fallback": False, "error": None,
        })
        response = client.post("/ask", json={"query": "quiero sacar pasaporte"})
    assert response.status_code == 200
    data = response.json()
    assert data["tramite"] == "Pasaporte"
    assert data["fuente"] is not None

def test_ask_cedula_keyword():
    with patch("main.call_llm") as mock_llm:
        mock_llm.return_value = ("Respuesta mock cedula", {
            "latency_ms": 100, "ttft_ms": 40, "input_tokens": 10,
            "output_tokens": 20, "total_tokens": 30,
            "fallback": False, "error": None,
        })
        response = client.post("/ask", json={"query": "renovar carnet"})
    assert response.status_code == 200
    data = response.json()
    assert data["tramite"] == "Cedula de identidad para chilenos"

def test_ask_fuera_dominio():
    response = client.post("/ask", json={"query": "quiero renovar licencia de conducir"})
    assert response.status_code == 200
    data = response.json()
    assert data["tramite"] == "Fuera del alcance de ARCO"

def test_ask_no_identificado():
    response = client.post("/ask", json={"query": "xyz consulta inexistente 12345"})
    assert response.status_code == 200
    data = response.json()
    assert data["tramite"] == "No identificado"

def test_respuesta_tiene_campos_obligatorios():
    with patch("main.call_llm") as mock_llm:
        mock_llm.return_value = ("Respuesta mock", {
            "latency_ms": 100, "ttft_ms": 40, "input_tokens": 10,
            "output_tokens": 20, "total_tokens": 30,
            "fallback": False, "error": None,
        })
        response = client.post("/ask", json={"query": "quiero sacar pasaporte"})
    data = response.json()
    campos = ["tramite", "respuesta", "costo", "duracion",
              "canal", "presencialidad", "requiere_clave_unica",
              "fuente", "trace_id", "model_used"]
    for campo in campos:
        assert campo in data, f"Campo '{campo}' falta en la respuesta"