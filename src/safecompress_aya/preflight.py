"""Stage A environment audit. This module never downloads model weights."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub",
            "circuitkit", "curatorkit", "auditkit", "numpy", "PyYAML")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_lock() -> str:
    """Snapshot installed distributions without invoking pip or exposing environment variables."""
    lines = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                   if d.metadata.get("Name"))
    return "\n".join(lines) + "\n"


def command_result(argv: list[str], timeout: int = 15) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                check=False)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()[:1000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def audit(config_path: Path, output_root: Path) -> Path:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict) or config.get("model_id") != "CohereLabs/tiny-aya-global":
        raise ValueError("Stage A expects the preregistered Tiny Aya model_id")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    lock = package_lock().encode("utf-8")
    (run_dir / "packages.lock.txt").write_bytes(lock)
    modules = {name: {"importable": importlib.util.find_spec(name.lower().replace("pyyaml", "yaml")) is not None,
                      "version": _version(name)} for name in PACKAGES}
    gpu = command_result(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"])
    torch_probe = command_result([sys.executable, "-c",
                                  "import json,torch; print(json.dumps({'cuda':torch.cuda.is_available(),'bf16':torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,'torch':torch.__version__}))"])
    cache = _model_config_cached(config["model_id"])
    report = {
        "run_id": run_id, "stage": "A_environment_preflight", "status": "preflight_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model_id"], "model_config_cached": cache,
        "model_access_verified": False, "quantization_save_reload_generation_verified": False,
        "python": platform.python_version(), "platform": platform.platform(),
        "config_sha256": sha256(config_bytes), "package_lock_sha256": sha256(lock),
        "packages": modules, "gpu": gpu, "torch_probe": torch_probe,
        "next_gate": "BF16 generation and uniform 4-bit save/reload/generation on authorized model weights",
    }
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (run_dir / "report.json").write_bytes(report_bytes)
    (run_dir / "SHA256SUMS").write_text(
        f"{sha256(lock)}  packages.lock.txt\n{sha256(report_bytes)}  report.json\n",
        encoding="utf-8")
    return run_dir


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _model_config_cached(model_id: str) -> bool:
    if importlib.util.find_spec("huggingface_hub") is None:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False
