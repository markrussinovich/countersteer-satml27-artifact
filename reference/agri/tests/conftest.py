from pathlib import Path
import sys


VENDORED_AGENTDOJO_SRC = Path(__file__).resolve().parents[1] / "vendor" / "agentdojo" / "src"
if VENDORED_AGENTDOJO_SRC.exists() and str(VENDORED_AGENTDOJO_SRC) not in sys.path:
    sys.path.insert(0, str(VENDORED_AGENTDOJO_SRC))
