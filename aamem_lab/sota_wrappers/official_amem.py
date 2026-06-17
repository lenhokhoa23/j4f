from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


class OfficialAMemWrapper:
    """Thin optional adapter around the official A-MEM package.

    This wrapper is intentionally not used by the default smoke tests because
    the upstream package may require ChromaDB, embedding models, and an LLM
    backend. It exists so Phase 2 can compare against the actual implementation
    without changing the experiment runner interface.
    """

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path).resolve()
        self.memory_system_cls: Optional[Any] = None
        self.import_error: Optional[BaseException] = None
        self._try_import()

    @property
    def available(self) -> bool:
        return self.memory_system_cls is not None

    def is_available(self) -> bool:
        return self.available

    def _try_import(self) -> None:
        try:
            sys.path.insert(0, str(self.repo_path))
            from agentic_memory.memory_system import AgenticMemorySystem  # type: ignore

            self.memory_system_cls = AgenticMemorySystem
        except BaseException as exc:  # pragma: no cover - depends on external deps
            self.import_error = exc
            self.memory_system_cls = None
        finally:
            try:
                sys.path.remove(str(self.repo_path))
            except ValueError:
                pass

    def explain_status(self) -> str:
        if self.available:
            return "Official A-MEM import is available."
        return f"Official A-MEM unavailable: {self.import_error!r}"
