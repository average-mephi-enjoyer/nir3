"""Project-local model identity and offline deployment verification."""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/local_llm.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Deployment path escapes local_llm")
    return path


def verify_deployment(root=None):
    root = Path(root) if root is not None else PROJECT_ROOT / "local_llm"
    config = load_config()
    receipt_path = root / "deployment.json"
    if not receipt_path.exists():
        raise RuntimeError("Local LLM is not installed. Run scripts/setup_local_llm.ps1 -Download manually.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if not receipt.get("ready") or receipt.get("config_sha256") != file_sha256(CONFIG_PATH):
        raise RuntimeError("Local deployment receipt/config mismatch; rerun setup verification")
    if receipt.get("model_name") != config["model_name"] or receipt.get("model_revision") != config["model_revision"]:
        raise RuntimeError("Unexpected local model identity")
    for artifact in config["artifacts"]:
        path = contained_path(root, artifact["path"])
        if not path.is_file() or path.stat().st_size != artifact["size"] or file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Missing or modified deployment artifact: {artifact['path']}")
    if not receipt.get("runtime_files"):
        raise RuntimeError("Runtime checksums are missing")
    for entry in receipt["runtime_files"]:
        path = contained_path(root, entry["path"])
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise RuntimeError("Portable runtime changed after installation")
    return {"model_repository": config["model_repository"], "model_revision": config["model_revision"],
            "runtime_tag": config["runtime_tag"], "receipt_sha256": file_sha256(receipt_path),
            "weight_sha256": [a["sha256"] for a in config["artifacts"] if a["kind"] == "model"],
            "verification": "local files hashed; server alias checked separately"}
