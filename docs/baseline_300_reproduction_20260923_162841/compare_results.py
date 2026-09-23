"""Recompute historical/fresh scores with both historical and canonical truth."""
import csv
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation_common import FIELDS, normalize_truth, score_prediction


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


labels = {r["file_name"]: r for r in read(ROOT / "labels/labels_300.csv")}
old = {r["file_name"]: r for r in read(ROOT / "outputs/ppocrv6_medium_300/v6_medium_300_results.csv")}
fresh = {r["file_name"]: r for r in read(OUT / "evaluation/evaluation_300.csv")}
assert len(labels) == len(old) == len(fresh) == 300 and labels.keys() == old.keys() == fresh.keys()
rows = []
for name, label in labels.items():
    a, b = old[name], fresh[name]
    truth = normalize_truth(label)
    previous = a["final_date_correct"].lower() == "true"
    rescored = score_prediction(a, truth)["final_date_correct"]
    current = score_prediction(b, truth)["final_date_correct"]
    assert current == (b["final_date_correct"].lower() == "true")
    rows.append({"image_id": label["image_id"], "file_name": name,
                 "historical_prediction": a["final_date"], "fresh_prediction": b["final_date"],
                 "historical_truth": a["true_final_date"], "canonical_truth": truth["final_date"],
                 "historical_correct": previous, "historical_canonical_correct": rescored,
                 "fresh_historical_truth_correct": b["final_date"] == a["true_final_date"],
                 "fresh_canonical_correct": current,
                 "normalization_changed_verdict": previous != rescored,
                 "inference_changed_verdict": rescored != current,
                 "prediction_changed": any(a[k] != b[k] for k in FIELDS),
                 "historical_method": a["method"], "fresh_method": b["method"]})
for filename, subset in [("image_comparison.csv", rows),
                         ("changed_images.csv", [r for r in rows if r["prediction_changed"] or r["normalization_changed_verdict"]])]:
    with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(subset)
summary = json.loads((OUT / "evaluation/summary.json").read_text(encoding="utf-8"))
execution = json.loads((OUT / "execution_meta.json").read_text(encoding="utf-8"))
before = json.loads((OUT / "protected_before.json").read_text(encoding="utf-8"))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in before.items())
result = {"count": 300,
          "historical_correct": sum(r["historical_correct"] for r in rows),
          "historical_predictions_canonical_correct": sum(r["historical_canonical_correct"] for r in rows),
          "fresh_predictions_historical_correct": sum(r["fresh_historical_truth_correct"] for r in rows),
          "fresh_correct": sum(r["fresh_canonical_correct"] for r in rows),
          "fresh_accuracy": sum(r["fresh_canonical_correct"] for r in rows) / 300,
          "normalization_changed_ids": [r["image_id"] for r in rows if r["normalization_changed_verdict"]],
          "prediction_changed_ids": [r["image_id"] for r in rows if r["prediction_changed"]],
          "inference_changed_verdict_ids": [r["image_id"] for r in rows if r["inference_changed_verdict"]],
          "init_sec": summary["init_sec"], "infer_sec": summary["infer_sec"],
          "avg_infer_sec_per_image": summary["avg_sec_per_image"], **execution,
          "historical_runtime_sec": 618.895585299997,
          "historical_timing_scope": "Inference loop only, after model initialization; not external wall time"}
(OUT / "comparison_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
