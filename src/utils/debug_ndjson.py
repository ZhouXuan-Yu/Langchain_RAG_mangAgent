# region agent log
"""Session 16253e — NDJSON debug sink (do not log secrets)."""

import json
import time
from pathlib import Path

_DEBUG_LOG = Path(__file__).resolve().parent.parent.parent / "debug-16253e.log"
_SESSION = "16253e"


def debug_ndjson(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        rec = {
            "sessionId": _SESSION,
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion
