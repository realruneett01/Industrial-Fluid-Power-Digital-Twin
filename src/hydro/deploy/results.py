"""Shared result types for the deployment gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


class Status(IntEnum):
    """Ordered outcome: the worst status of a step decides the gate."""

    PASS = 0
    WARN = 1
    FAIL = 2

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class CheckResult:
    """One named check with an outcome, a detail line and an optional fix hint."""

    name: str
    status: Status
    detail: str
    hint: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.name
        return out


def worst(results: list[CheckResult]) -> Status:
    """Worst status in results; an empty list is a PASS."""
    return max((r.status for r in results), default=Status.PASS)


def render(results: list[CheckResult]) -> str:
    """Plain-text table for terminals (ASCII only, safe on legacy Windows consoles)."""
    lines = []
    for r in results:
        lines.append(f"[{r.status.name:<4}] {r.name}: {r.detail}")
        if r.hint and r.status != Status.PASS:
            lines.append(f"       -> {r.hint}")
    return "\n".join(lines)
