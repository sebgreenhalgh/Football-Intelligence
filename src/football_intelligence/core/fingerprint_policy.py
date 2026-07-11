from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


DEFAULT_RUNTIME_FIELD_NAMES = frozenset(
    {
        "created_at",
        "completed_at",
        "started_at",
        "updated_at",
        "runtime_hostname",
        "process_id",
        "execution_duration_seconds",
    }
)


def _json_path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    if parent == "$":
        return f"$.{key}"
    return f"{parent}.{key}"


@dataclass(frozen=True)
class SemanticFingerprintPolicy:
    excluded_field_names: frozenset[str] = DEFAULT_RUNTIME_FIELD_NAMES
    excluded_json_paths: frozenset[str] = field(default_factory=frozenset)
    set_like_json_paths: frozenset[str] = field(default_factory=frozenset)

    def canonicalize(self, value: Any, path: str = "$") -> Any:
        if path in self.excluded_json_paths:
            return None
        if isinstance(value, dict):
            canonical: dict[str, Any] = {}
            for key in sorted(value):
                key_text = str(key)
                child_path = _json_path(path, key_text)
                if key_text in self.excluded_field_names or child_path in self.excluded_json_paths:
                    continue
                canonical[key_text] = self.canonicalize(value[key], child_path)
            return canonical
        if isinstance(value, list):
            rows = [self.canonicalize(item, _json_path(path, index)) for index, item in enumerate(value)]
            if path in self.set_like_json_paths:
                return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
            return rows
        if isinstance(value, tuple):
            return self.canonicalize(list(value), path)
        return value


DEFAULT_POLICY = SemanticFingerprintPolicy()


def canonical_semantic_json(value: Any, policy: SemanticFingerprintPolicy | None = None) -> str:
    active_policy = policy or DEFAULT_POLICY
    canonical = active_policy.canonicalize(value)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
