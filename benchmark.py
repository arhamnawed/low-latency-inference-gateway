import sys
from pathlib import Path

gateway_dir = Path(__file__).resolve().parent / "inference_gateway"
sys.path.insert(0, str(gateway_dir))

from inference_gateway.benchmark import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        total_requests=1000,
        duplication_rate=0.35,
        concurrency_workers=16,
        max_batch_size=32,
        batch_timeout_ms=8.0,
    )
