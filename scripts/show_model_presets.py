from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aamem_lab.model_presets import presets_as_dict


def main() -> None:
    print(json.dumps(presets_as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
