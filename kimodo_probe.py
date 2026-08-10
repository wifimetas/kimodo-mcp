# SPDX-License-Identifier: Apache-2.0
"""Environment probe run inside the Kimodo virtualenv. Prints one JSON object."""

from __future__ import annotations

import json
import sys

d = {"python": sys.version.split()[0], "executable": sys.executable}

try:
    import torch

    d["torch"] = torch.__version__
    d["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        d["vram_free_gb"] = round(free / 1024 ** 3, 2)
        d["vram_total_gb"] = round(total / 1024 ** 3, 2)
except Exception as exc:  # noqa: BLE001
    d["torch_error"] = str(exc)

try:
    import kimodo

    d["kimodo_package"] = getattr(kimodo, "__file__", "?")
    from kimodo.model.registry import AVAILABLE_MODELS, DEFAULT_MODEL

    d["available_models"] = list(AVAILABLE_MODELS)
    d["kimodo_default_model"] = DEFAULT_MODEL
except Exception as exc:  # noqa: BLE001
    d["kimodo_error"] = str(exc)

try:
    from kimodo.demo.memory_manager import manager  # noqa: F401

    d["offload_supported"] = True
except Exception:
    d["offload_supported"] = False

try:
    from kimodo.demo.embedding_cache import CachedTextEncoder  # noqa: F401

    d["embedding_cache"] = True
except Exception:
    d["embedding_cache"] = False

print(json.dumps(d))
