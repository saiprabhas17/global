"""Download the configured local models before serving NLP requests.

Run from backend: python scripts/prepare_models.py
The output records resolved snapshot revisions for reproducible deployments.
spaCy is installed separately: python -m spacy download en_core_web_sm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-only", action="store_true", help="Check cached snapshots without network access"
    )
    parser.add_argument(
        "--only", choices=("embedding", "extraction", "stance"),
        help="Prepare one capability instead of all three",
    )
    args = parser.parse_args()
    settings = get_settings()
    embedding = settings.EMBEDDING_MODEL
    if "/" not in embedding:
        embedding = "sentence-transformers/" + embedding
    models = {
        "embedding": (embedding, None),
        "extraction": (os.environ.get("HF_LLM_MODEL", "google/flan-t5-base"), None),
        "stance": (settings.NLI_MODEL, settings.NLI_MODEL_REVISION),
    }
    manifest = []
    for capability, (model, revision) in models.items():
        if args.only and capability != args.only:
            continue
        if Path(model).is_dir():
            snapshot = Path(model).resolve()
            resolved_revision = None
        else:
            snapshot = Path(snapshot_download(
                model,
                revision=revision,
                local_files_only=args.local_only,
                # Prefer safetensors; do not download duplicate PyTorch/ONNX/TF weights.
                allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
                max_workers=3,
            ))
            resolved_revision = snapshot.name
        if not list(snapshot.glob("*.safetensors")):
            raise RuntimeError(f"No safetensors weights cached for {capability}: {model}")
        manifest.append({
            "capability": capability, "model": model,
            "revision": resolved_revision, "snapshot": str(snapshot),
        })
    print(json.dumps({"models": manifest, "inference_verified": False}, indent=2))


if __name__ == "__main__":
    main()
