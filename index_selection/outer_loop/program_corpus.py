"""
Program corpus management for the outer loop.

Manages the set of baseline/evolved programs used to validate
evaluation strategies via ranking agreement.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ProgramInfo:
    """Metadata about a program in the corpus."""
    name: str
    path: str
    category: str  # "baseline" or "evolved"


class ProgramCorpus:
    """Manages the fixed set of programs for ranking validation."""

    def __init__(self, program_paths: List[str]):
        self.programs: List[ProgramInfo] = []
        for path in program_paths:
            if not os.path.exists(path):
                print(f"Warning: program not found, skipping: {path}")
                continue
            name = Path(path).stem
            category = "evolved" if name.startswith("best_") else "baseline"
            self.programs.append(ProgramInfo(name=name, path=path, category=category))

        if len(self.programs) < 2:
            raise ValueError(
                f"Need at least 2 programs for ranking comparison, "
                f"found {len(self.programs)}"
            )

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.programs]

    @property
    def paths(self) -> List[str]:
        return [p.path for p in self.programs]

    def get_path(self, name: str) -> str:
        for p in self.programs:
            if p.name == name:
                return p.path
        raise KeyError(f"Program not found: {name}")

    def __len__(self) -> int:
        return len(self.programs)

    def __repr__(self) -> str:
        return f"ProgramCorpus({len(self.programs)} programs: {self.names})"
