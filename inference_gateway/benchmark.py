from __future__ import annotations
import concurrent.futures
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.pipeline import InferencePipeline, InferenceResult, FastMLBackend


def generate_benchmark_workload(
    total_requests: int = 1000,
    duplication_rate: float = 0.35,
) -> Tuple[List[str], int, int]:
    num_duplicates = int(total_requests * duplication_rate)
    num_unique = total_requests - num_duplicates

    templates = [
        "This product is {adj} and {adj2}!",
        "The system latency is {adj}, absolutely {adj2}.",
        "Customer support was {adj} and remarkably {adj2}.",
        "I found this tool to be {adj}, truly {adj2}.",
        "Performance under load is {adj}, definitely {adj2}.",
        "The new update is {adj} with {adj2} quality.",
        "Architecture design is {adj} and codebase is {adj2}.",
        "Overall experience has been {adj} and {adj2}.",
    ]
    positive_adjs = ["amazing", "excellent", "superb", "fast", "reliable", "fantastic", "great", "smooth"]
    negative_adjs = ["terrible", "awful", "horrible", "slow", "broken", "buggy", "frustrating", "laggy"]

    unique_corpus: List[str] = []
    for i in range(num_unique):
        tmpl = templates[i % len(templates)]
        if i % 2 == 0:
            adj1 = positive_adjs[(i * 3) % len(positive_adjs)]
            adj2 = positive_adjs[(i * 7 + 1) % len(positive_adjs)]
        else:
            adj1 = negative_adjs[(i * 3) % len(negative_adjs)]
            adj2 = negative_adjs[(i * 7 + 1) % len(negative_adjs)]
        unique_corpus.append(f"[{i:04d}] " + tmpl.format(adj=adj1, adj2=adj2))

    random.seed(42)
    duplicate_corpus = random.choices(unique_corpus[:max(1, len(unique_corpus) // 3)], k=num_duplicates)

    full_workload: List[str] = []
    u_idx = 0
    d_idx = 0
    while u_idx < len(unique_corpus) or d_idx < len(duplicate_corpus):
        if d_idx < len(duplicate_corpus) and (random.random() < duplication_rate or u_idx >= len(unique_corpus)):
            full_workload.append(duplicate_corpus[d_idx])
            d_idx += 1
        elif u_idx < len(unique_corpus):
            full_workload.append(unique_corpus[u_idx])
            u_idx += 1

    return full_workload[:total_requests], num_unique, num_duplicates


def run_benchmark(
    total_requests: int = 1000,
    duplication_rate: float = 0.35,
    concurrency_workers: int = 16,
    max_batch_size: int = 32,
    batch_timeout_ms: float = 8.0,
) -> None:
    print("=" * 80)
    print("  LOW-LATENCY ML INFERENCE GATEWAY - PERFORMANCE BENCHMARK")
    print("=" * 80)

    workload, num_unique, num_duplicates = generate_benchmark_workload(
        total_requests=total_requests,
        duplication_rate=duplication_rate,
    )

    print(f"\n[Workload Configuration]")
    print(f"  - Total Ingress Requests:     {len(workload):,}")
    print(f"  - Unique Query Keys:          {num_unique:,} ({((num_unique/len(workload))*100):.1f}%)")
    print(f"  - Duplicate Query Keys:       {num_duplicates:,} ({((num_duplicates/len(workload))*100):.1f}%)")
    print(f"  - Gateway Concurrency:        {concurrency_workers} Parallel Worker Threads")
    print(f"  - Dynamic Batch Max Size:     {max_batch_size}")
    print(f"  - Dynamic Batch Timeout:      {batch_timeout_ms} ms")

    backend = FastMLBackend()
    pipeline = InferencePipeline(
        model=backend,
        cache_capacity=5000,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        metrics_buffer_size=10000,
    )

    results: List[InferenceResult] = []
    print(f"\n[Execution] Dispatching {len(workload)} concurrent queries through gateway...")

    t0_wall = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_workers) as executor:
        futures = [executor.submit(pipeline.predict, query) for query in workload]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    t_total_sec = time.perf_counter() - t0_wall

    pipeline.stop()

    hit_results = [r for r in results if r.cached]
    miss_results = [r for r in results if not r.cached]

    hit_count = len(hit_results)
    miss_count = len(miss_results)
    actual_hit_ratio = (hit_count / len(results)) * 100.0 if results else 0.0

    hit_latencies = [r.latency_ms for r in hit_results]
    miss_latencies = [r.latency_ms for r in miss_results]
    all_latencies = [r.latency_ms for r in results]

    mean_hit_lat = (sum(hit_latencies) / len(hit_latencies)) if hit_latencies else 0.0
    mean_miss_lat = (sum(miss_latencies) / len(miss_latencies)) if miss_latencies else 0.0
    mean_all_lat = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0.0

    overall_pct = pipeline.overall_metrics.compute_percentiles([50.0, 90.0, 95.0, 99.0])
    hit_pct = pipeline.hit_metrics.compute_percentiles([50.0, 95.0, 99.0])
    miss_pct = pipeline.miss_metrics.compute_percentiles([50.0, 95.0, 99.0])

    throughput_rps = len(results) / t_total_sec if t_total_sec > 0 else 0.0
    speedup = (mean_miss_lat / mean_hit_lat) if mean_hit_lat > 0 else 0.0

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Wall-Clock Duration:          {t_total_sec * 1000.0:.2f} ms ({t_total_sec:.3f} s)")
    print(f"  Gateway Throughput:           {throughput_rps:,.2f} Requests/sec (RPS)")
    print(f"  Cache Hit Ratio:              {actual_hit_ratio:.2f}% (Target: ~{duplication_rate*100:.1f}%)")
    print(f"  Cache Hits (Fast Path):       {hit_count:,}")
    print(f"  Cache Misses (Slow Path):     {miss_count:,}")
    print(f"  Cache Speedup Factor:         {speedup:.1f}x faster on Cache Hit")

    print("\n" + "-" * 80)
    print(f"{'Path':<24} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10}")
    print("-" * 80)
    print(f"{'Cache Hit (Fast Path)':<24} | {mean_hit_lat:<10.4f} | {hit_pct.get('p50', 0.0):<10.4f} | {hit_pct.get('p95', 0.0):<10.4f} | {hit_pct.get('p99', 0.0):<10.4f}")
    print(f"{'Cache Miss (Slow Path)':<24} | {mean_miss_lat:<10.4f} | {miss_pct.get('p50', 0.0):<10.4f} | {miss_pct.get('p95', 0.0):<10.4f} | {miss_pct.get('p99', 0.0):<10.4f}")
    print(f"{'End-to-End Gateway':<24} | {mean_all_lat:<10.4f} | {overall_pct.get('p50', 0.0):<10.4f} | {overall_pct.get('p95', 0.0):<10.4f} | {overall_pct.get('p99', 0.0):<10.4f}")
    print("-" * 80)

    stats = pipeline.stats()
    print("\n[Underlying DSA Engine Telemetry]")
    print(f"  - LRU Cache Lookups:          {stats['cache']['hits'] + stats['cache']['misses']:,}")
    print(f"  - LRU Cache Evictions:        {stats['cache']['evictions']:,}")
    print(f"  - Batches Dispatched:         {stats['batches_processed']:,}")
    print(f"  - Average Batch Size:         {stats['avg_batch_size']} items/batch")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark(
        total_requests=1000,
        duplication_rate=0.35,
        concurrency_workers=16,
        max_batch_size=32,
        batch_timeout_ms=8.0,
    )
