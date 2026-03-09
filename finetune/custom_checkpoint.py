from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Tuple

EXPORT_VERSION = 1
EXPORT_METADATA_FILE = "nayhein_export.json"
REMOTE_CONFIG_FILE = "configuration_nayhein_mini.py"
REMOTE_MODEL_FILE = "modeling_nayhein_mini.py"


def is_custom_checkpoint_dir(path: str | Path) -> bool:
    checkpoint_dir = Path(path)
    return checkpoint_dir.is_dir() and (checkpoint_dir / "model.pt").exists() and (checkpoint_dir / "config.pt").exists()


def get_hf_export_dir(path: str | Path) -> Path:
    return Path(path) / "hf_format"


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _has_tokenizer_files(export_dir: Path) -> bool:
    if (export_dir / "tokenizer_config.json").exists():
        if (export_dir / "tokenizer.json").exists():
            return True
        if (export_dir / "vocab.json").exists() and (export_dir / "merges.txt").exists():
            return True
    return False


def is_hf_export_stale(checkpoint_dir: Path, export_dir: Path) -> Tuple[bool, str]:
    required_files = [
        export_dir / "config.json",
        export_dir / "pytorch_model.bin",
        export_dir / REMOTE_CONFIG_FILE,
        export_dir / REMOTE_MODEL_FILE,
        export_dir / EXPORT_METADATA_FILE,
    ]
    for required in required_files:
        if not required.exists():
            return True, f"missing {required.name}"

    if not _has_tokenizer_files(export_dir):
        return True, "missing tokenizer files"

    metadata_path = export_dir / EXPORT_METADATA_FILE
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return True, f"invalid export metadata: {exc}"

    if metadata.get("export_version") != EXPORT_VERSION:
        return True, "export version mismatch"

    for name in ("model.pt", "config.pt"):
        source_path = checkpoint_dir / name
        source_fp = _file_fingerprint(source_path)
        recorded = metadata.get("source_files", {}).get(name)
        if recorded != source_fp:
            return True, f"source checkpoint changed: {name}"

    return False, ""


def ensure_hf_export(
    checkpoint_dir: str | Path,
    export_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    if not is_custom_checkpoint_dir(checkpoint_path):
        raise FileNotFoundError(
            f"Custom checkpoint not found at {checkpoint_path}. Expected model.pt and config.pt."
        )

    target_dir = Path(export_dir) if export_dir is not None else get_hf_export_dir(checkpoint_path)

    stale_reason = ""
    if force:
        stale_reason = "forced rebuild"
    else:
        is_stale, stale_reason = is_hf_export_stale(checkpoint_path, target_dir)
        if not is_stale:
            return target_dir

    print(f"Regenerating HF export at {target_dir} ({stale_reason})")
    from tools.export_hf_format import export_dualmode_to_hf

    export_dualmode_to_hf(checkpoint_path, target_dir)
    return target_dir


def clear_hf_module_cache(export_dir: str | Path) -> None:
    module_name = Path(export_dir).name
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / module_name
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
