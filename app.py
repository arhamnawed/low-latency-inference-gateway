import sys
from pathlib import Path

gateway_dir = Path(__file__).resolve().parent / "inference_gateway"
if str(gateway_dir) not in sys.path:
    sys.path.insert(0, str(gateway_dir))

from inference_gateway import app
