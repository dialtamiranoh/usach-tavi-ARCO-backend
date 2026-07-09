"""
eval/evaluador.py
Evaluacion automatica de ARCO con ground truth.
Corre sin LLM — solo valida keyword search.
Uso: python eval/evaluador.py
"""

import os
import json
import sys
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

os.environ["USE_RAG"] = "false"
os.environ["ARCO_SKIP_MODEL_LOAD"] = "true"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from main import app

CASOS_PATH = Path(__file__).resolve().parent / "casos_prueba.json"



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

REPORTE_PATH = Path(__file__).resolve().parent / f"reporte_{timestamp}.json"
REPORTE_LATEST = Path(__file__).resolve().parent / "reporte_latest.json"

ACCURACY_MINIMA = 0.90

client = TestClient(app)

MOCK_META = {
    "latency_ms": 0, "ttft_ms": None, "input_tokens": 0, "output_tokens": 0,
    "total_tokens": 0, "fallback": False, "error": None,
}

def evaluar():
    with open(CASOS_PATH, "r", encoding="utf-8") as f:
        casos = json.load(f)

    resultados = []
    correctos = 0

    print(f"\n{'='*60}")
    print(f"ARCO — Evaluación automática con ground truth")
    print(f"{'='*60}")
    print(f"Total casos: {len(casos)}")
    print(f"Accuracy mínima requerida: {ACCURACY_MINIMA*100:.0f}%")
    print(f"{'='*60}\n")

    for caso in casos:

        query_norm = caso["query"].lower()
        for item in __import__('json').load(open('knowledge.json', encoding='utf-8')):
            for kw in item["keywords"]:
                if kw.lower() in query_norm:
                    print(f"  DEBUG [{caso['id']}] match: '{kw}' → {item['id']}")
                    break
            else:
                continue
            break
        
        with patch("main.call_llm") as mock_llm:
            mock_llm.return_value = ("respuesta mock", MOCK_META)
            response = client.post("/ask", json={"query": caso["query"]})

        data = response.json()
        tramite_obtenido = data.get("tramite", "")
        tramite_correcto = tramite_obtenido == caso["tramite_esperado"]

        respuesta_texto = (
            (data.get("respuesta") or "") + " " +
            (data.get("respuesta_base") or "")
        ).lower()
        keywords_ok = all(
            kw.lower() in respuesta_texto
            for kw in caso.get("keywords_respuesta", [])
        )

        paso = tramite_correcto
        if paso:
            correctos += 1

        estado = "✅ PASS" if paso else "❌ FAIL"
        print(f"{estado} [{caso['id']}] {caso['query'][:45]}")
        if not tramite_correcto:
            print(f"       Esperado:  {caso['tramite_esperado']}")
            print(f"       Obtenido:  {tramite_obtenido}")

        resultados.append({
            "id": caso["id"],
            "query": caso["query"],
            "tramite_esperado": caso["tramite_esperado"],
            "tramite_obtenido": tramite_obtenido,
            "tramite_correcto": tramite_correcto,
            "keywords_ok": keywords_ok,
            "paso": paso
        })

    accuracy = correctos / len(casos)
    print(f"\n{'='*60}")
    print(f"Resultados: {correctos}/{len(casos)} correctos")
    print(f"Accuracy:   {accuracy*100:.1f}%")
    print(f"Requerida:  {ACCURACY_MINIMA*100:.0f}%")
    print(f"Estado:     {'✅ APROBADO' if accuracy >= ACCURACY_MINIMA else '❌ REPROBADO'}")
    print(f"{'='*60}\n")

    reporte = {
        "total_casos": len(casos),
        "correctos": correctos,
        "accuracy": round(accuracy, 4),
        "accuracy_minima": ACCURACY_MINIMA,
        "aprobado": accuracy >= ACCURACY_MINIMA,
        "resultados": resultados
    }

    with open(REPORTE_PATH, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    with open(REPORTE_LATEST, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print(f"Reporte guardado en: {REPORTE_PATH}")
    print(f"Ultimo reporte:      {REPORTE_LATEST}")

    if accuracy < ACCURACY_MINIMA:
        sys.exit(1)

if __name__ == "__main__":
    evaluar()