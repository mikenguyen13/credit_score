"""Operational state: freeze flags and the human-in-the-loop kill switch.

Three load-bearing flags live here:

* macro_shock_freeze   set by an operator when the macro environment
                       breaks (COVID-style). Hard-blocks any auto-
                       promotion until cleared. The challenger still
                       trains and the artifact is written; only the
                       registry transition is blocked.

* bureau_outage        set when the bureau feed is degraded. Retrain
                       runs on the matured slice as usual but
                       predictions returned at runtime are flagged
                       so downstream consumers know the score is
                       computed against a stale label distribution.

* iv_kill              set when the validator manually disables the
                       Heckman path because the exclusion restriction
                       has failed conceptual review. Forces fall-back
                       to AIPW-only (or to bounds-only when AIPW also
                       fails the support check).

State is persisted to a JSON file so the orchestrator and the FastAPI
service share the same view without an external dependency. A real
deployment would back this with the model registry's tag system or a
small key-value store; the file is the minimal contract.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_DEFAULT_PATH = Path(os.environ.get(
    "RI_STATE_PATH", "/tmp/reject_inference_state.json"))


@dataclass
class OperationalState:
    """Persistent operational flags for the pipeline."""

    macro_shock_freeze: bool = False
    bureau_outage: bool = False
    iv_kill: bool = False
    macro_shock_reason: str = ""
    last_updated: str = ""
    last_updated_by: str = ""

    def freeze_active(self) -> bool:
        return self.macro_shock_freeze or self.iv_kill

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "OperationalState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def load_state(path: Path | str | None = None) -> OperationalState:
    p = Path(path) if path is not None else _DEFAULT_PATH
    if not p.exists():
        return OperationalState()
    try:
        return OperationalState.from_dict(json.loads(p.read_text()))
    except Exception:
        return OperationalState()


def save_state(state: OperationalState,
               path: Path | str | None = None,
               actor: str = "system") -> Path:
    p = Path(path) if path is not None else _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = datetime.now(timezone.utc).isoformat()
    state.last_updated_by = actor
    p.write_text(json.dumps(state.to_dict(), indent=2))
    return p


def set_macro_shock(active: bool, reason: str = "",
                    actor: str = "operator",
                    path: Path | str | None = None) -> OperationalState:
    state = load_state(path)
    state.macro_shock_freeze = active
    state.macro_shock_reason = reason if active else ""
    save_state(state, path, actor)
    return state


def set_bureau_outage(active: bool, actor: str = "operator",
                      path: Path | str | None = None) -> OperationalState:
    state = load_state(path)
    state.bureau_outage = active
    save_state(state, path, actor)
    return state


def set_iv_kill(active: bool, actor: str = "operator",
                path: Path | str | None = None) -> OperationalState:
    state = load_state(path)
    state.iv_kill = active
    save_state(state, path, actor)
    return state


@dataclass
class FreezeBlock:
    """Result of a pre-promotion freeze check."""

    blocked: bool
    reasons: list[str] = field(default_factory=list)


def check_freeze(state: Optional[OperationalState] = None,
                 path: Path | str | None = None) -> FreezeBlock:
    """Inspect operational state and emit a structured block decision."""
    s = state if state is not None else load_state(path)
    reasons: list[str] = []
    if s.macro_shock_freeze:
        reasons.append(
            f"macro shock freeze active: {s.macro_shock_reason or '(no reason)'}"
        )
    if s.iv_kill:
        reasons.append("IV kill switch active: validator disabled Heckman path")
    return FreezeBlock(blocked=bool(reasons), reasons=reasons)
