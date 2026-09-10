import sys
from pathlib import Path

# Make the package importable as `sensor.`, `perception.`, etc. when pytest
# is run from the repo root without an editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
