import json
from pathlib import Path

baseline = Path("eval/baseline.json")
latest = Path("eval/reporte_latest.json")

with open(baseline, encoding="utf-8") as f:
    b = json.load(f)

print(f"Baseline: {b['correctos']}/{b['total_casos']} correctos - accuracy {b['accuracy']*100:.1f}%")

if latest.exists():
    with open(latest, encoding="utf-8") as f:
        l = json.load(f)
    
    print(f"Actual:   {l['correctos']}/{l['total_casos']} correctos - accuracy {l['accuracy']*100:.1f}%")
    
    casos_baseline = {r['id']: r['paso'] for r in b['resultados']}
    regresiones = []
    for r in l['resultados']:
        if casos_baseline.get(r['id']) and not r['paso']:
            regresiones.append(r['id'])
    
    if regresiones:
        print(f"\n❌ REGRESION detectada en: {regresiones}")
        exit(1)
    else:
        print("\n✅ Sin regresiones detectadas")