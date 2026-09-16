# Low-Latency ML Inference Gateway

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

An asynchronous inference gateway for high-throughput, low-latency machine learning serving. Implements custom core data structures without third-party collections to guarantee strict latency bounds and thread safety.

## Architecture

```
                 Incoming Request
                        │
                        ▼
                SHA-256 Text Hash
                        │
                        ▼
                  LRU Cache
                   /     \
           (Hit)  /       \  (Miss)
                 /         \
                ▼           ▼
        Cached Result   Batch Queue
         (~0.007ms)         │
                            ▼
                     Dynamic Batcher
                    (size 32 / 8ms)
                            │
                            ▼
                     Model Inference
                            │
                     ┌──────┴──────┐
                     ▼             ▼
                Update Cache  Ring Buffer
                             (Metrics)
```

## Complexity Analysis

### LRU Cache (`dsa/lru_cache.py`)
Doubly linked list with dummy head/tail sentinels and a hash map. Thread safety via `threading.RLock`.

| Method | Time | Space | Notes |
| :--- | :---: | :---: | :--- |
| `get(key)` | O(1) | O(1) | Hash lookup and moves node to head |
| `put(key, value)` | O(1) | O(1) | Updates existing or evicts tail node |
| `peek(key)` | O(1) | O(1) | Inspection without order update |
| `get_keys_ordered()` | O(N) | O(N) | Returns keys from MRU to LRU for UI visualization |
| `contains(key)` | O(1) | O(1) | Key presence check |
| `clear()` | O(1) | O(1) | Resets internal map and sentinels |

### Batch Queue (`dsa/batch_queue.py`)
FIFO linked queue with explicit head and tail pointers and condition variable signaling.

| Method | Time | Space | Notes |
| :--- | :---: | :---: | :--- |
| `enqueue(item)` | O(1) | O(1) | Appends to tail and notifies waiting consumers |
| `dequeue(timeout)` | O(1) | O(1) | Extracts from head with optional timed wait |
| `collect_batch(max_batch_size, timeout_ms)` | O(K) | O(K) | Flushes up to max_batch_size items or waits until timeout |
| `clear()` | O(N) | O(1) | Traverses and clears linked nodes |

### Ring Buffer (`dsa/ring_buffer.py`)
Fixed-capacity circular array for latency tracking with a pre-allocated scratch buffer for in-place percentile computation.

| Method | Time | Space | Notes |
| :--- | :---: | :---: | :--- |
| `append(value)` | O(1) | O(1) | Writes to write_index % capacity, overwriting oldest entry |
| `mean()` | O(1) | O(1) | Rolling sum divided by sample count |
| `percentile(p)` | O(K log K) | O(1) | In-place partition sort on scratch array with linear interpolation |
| `compute_percentiles()` | O(K log K) | O(1) | Single sorting pass for p50, p95, and p99 |

## Project Structure

```
inference_gateway/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                 # Streamlit dashboard
├── benchmark.py           # Concurrency and throughput simulation
├── dsa/
│   ├── __init__.py
│   ├── lru_cache.py       # Doubly linked list + hash map LRU cache
│   ├── batch_queue.py     # FIFO queue with dynamic batch collection
│   └── ring_buffer.py     # Fixed-size circular array for latency percentiles
├── engine/
│   ├── __init__.py
│   └── pipeline.py        # Pipeline integrating cache, queue, and model inference
└── tests/
    ├── __init__.py
    └── test_dsa.py        # Unit and integration tests
```

## Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run tests
```bash
pytest tests/ -v
```

### 3. Run benchmark
```bash
python benchmark.py
```

### 4. Launch interactive dashboard
```bash
streamlit run app.py
```

## Benchmark Output

Results from a benchmark of 1,000 concurrent requests across 16 worker threads with 35% duplicate queries:

```
================================================================================
  LOW-LATENCY ML INFERENCE GATEWAY - PERFORMANCE BENCHMARK
================================================================================

[Workload Configuration]
  - Total Ingress Requests:     1,000
  - Unique Query Keys:          650 (65.0%)
  - Duplicate Query Keys:       350 (35.0%)
  - Gateway Concurrency:        16 Parallel Worker Threads
  - Dynamic Batch Max Size:     32
  - Dynamic Batch Timeout:      8.0 ms

[Execution] Dispatching 1000 concurrent queries through gateway...

================================================================================
  BENCHMARK RESULTS SUMMARY
================================================================================
  Wall-Clock Duration:          551.45 ms (0.551 s)
  Gateway Throughput:           1,813.41 Requests/sec (RPS)
  Cache Hit Ratio:              34.40% (Target: ~35.0%)
  Cache Hits (Fast Path):       344
  Cache Misses (Slow Path):     656
  Cache Speedup Factor:         1826.5x faster on Cache Hit

--------------------------------------------------------------------------------
Path                     | Mean (ms)  | p50 (ms)   | p95 (ms)   | p99 (ms)  
--------------------------------------------------------------------------------
Cache Hit (Fast Path)    | 0.0068     | 0.0031     | 0.0294     | 0.0443    
Cache Miss (Slow Path)   | 12.4152    | 12.6184    | 14.9010    | 21.3769   
End-to-End Gateway       | 8.1467     | 11.2169    | 14.5350    | 21.2415   
--------------------------------------------------------------------------------

[Underlying DSA Engine Telemetry]
  - LRU Cache Lookups:          1,000
  - LRU Cache Evictions:        0
  - Batches Dispatched:         41
  - Average Batch Size:         16.0 items/batch
================================================================================
```
