
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

files = [
    ROOT / "tests_" / "atmosync_predict.json",
    ROOT / "models" / "atmosync_tcn_missing_robust"
         / "predictions" / "final_test_predictions.csv",
]

for path in files:
    print("\n" + "=" * 75)
    print("FILE:", path)
    print("EXISTS:", path.exists())

    if not path.exists():
        continue

    if path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            print("TOP-LEVEL TYPE:", type(obj).__name__)

            if isinstance(obj, dict):
                print("TOP-LEVEL KEYS:", list(obj.keys()))

            print("\nJSON CONTENT (first 12000 characters):")
            print(json.dumps(obj, indent=2, default=str)[:12000])

        except Exception as exc:
            print("JSON READ ERROR:", repr(exc))

    elif path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
            print("SHAPE:", df.shape)
            print("COLUMNS:", df.columns.tolist())
            print("\nFIRST 10 ROWS:")
            print(df.head(10).to_string(index=False))

        except Exception as exc:
            print("CSV READ ERROR:", repr(exc))
