from pathlib import Path
import csv
from sklearn.datasets import load_diabetes

OUT = Path(__file__).resolve().parent / "data" / "diabetes_sklearn.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

data = load_diabetes(as_frame=False, scaled=False)
headers = list(data.feature_names) + ["target"]

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for row, target in zip(data.data, data.target):
        writer.writerow([float(v) for v in row] + [float(target)])

print(f"CSV generado: {OUT}")
print(f"Registros: {len(data.data)}")
