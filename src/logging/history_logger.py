from __future__ import annotations

import csv
from pathlib import Path


class HistoryLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def append(self, row: dict):
        self.rows.append(dict(row))
        fieldnames = []
        for item in self.rows:
            for key in item:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
