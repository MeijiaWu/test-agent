import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_tasks(path, split="all", ids=None, limit=None):
    tasks = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if split != "all":
        tasks = [t for t in tasks if t.get("split") == split]
    if ids:
        tasks = [t for t in tasks if t["instance_id"] in set(ids)]
    return tasks[:limit] if limit else tasks


def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
