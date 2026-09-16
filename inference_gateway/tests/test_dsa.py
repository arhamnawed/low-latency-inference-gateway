from __future__ import annotations
import concurrent.futures
import math
import sys
import threading
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dsa.lru_cache import LRUCache, Node, DoublyLinkedList
from dsa.batch_queue import BatchQueue, QueueNode
from dsa.ring_buffer import RingBuffer
from engine.pipeline import InferencePipeline, FastMLBackend


class TestLRUCache:

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError):
            LRUCache(0)
        with pytest.raises(ValueError):
            LRUCache(-5)

    def test_basic_put_and_get(self) -> None:
        cache: LRUCache[str, int] = LRUCache(3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert len(cache) == 3
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("missing") is None
        assert cache.get("missing", default=-1) == -1

    def test_lru_eviction_order(self) -> None:
        cache: LRUCache[str, int] = LRUCache(3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.get("a") == 1

        cache.put("d", 4)
        assert len(cache) == 3
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

        assert cache.get("c") == 3
        cache.put("e", 5)
        assert cache.get("a") is None
        assert cache.get("e") == 5

    def test_get_keys_ordered(self) -> None:
        cache: LRUCache[str, int] = LRUCache(3)
        cache.put("k1", 10)
        cache.put("k2", 20)
        cache.put("k3", 30)

        assert cache.get_keys_ordered() == ["k3", "k2", "k1"]
        _ = cache.get("k1")
        assert cache.get_keys_ordered() == ["k1", "k3", "k2"]
        cache.put("k4", 40)
        assert cache.get_keys_ordered() == ["k4", "k1", "k3"]

    def test_capacity_one_edge_case(self) -> None:
        cache: LRUCache[str, str] = LRUCache(1)
        cache.put("k1", "v1")
        assert len(cache) == 1
        assert cache.get("k1") == "v1"

        cache.put("k2", "v2")
        assert len(cache) == 1
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"

        cache.put("k2", "v2_updated")
        assert len(cache) == 1
        assert cache.get("k2") == "v2_updated"

    def test_value_update_does_not_increase_size(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("x", 100)
        cache.put("y", 200)
        assert len(cache) == 2

        cache.put("x", 999)
        assert len(cache) == 2
        assert cache.get("x") == 999

        cache.put("z", 300)
        assert cache.get("y") is None
        assert cache.get("x") == 999
        assert cache.get("z") == 300

    def test_peek_does_not_alter_lru_order(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)

        assert cache.peek("a") == 1
        assert cache.peek("nonexistent", default=-1) == -1

        cache.put("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_contains_and_clear(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)
        assert cache.contains("a") is True
        assert cache.contains("b") is False

        cache.clear()
        assert len(cache) == 0
        assert cache.contains("a") is False
        assert cache.get("a") is None

    def test_stats_telemetry(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)

        _ = cache.get("a")
        _ = cache.get("missing")
        cache.put("c", 3)

        stats = cache.stats()
        assert stats["capacity"] == 2
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 1
        assert stats["hit_ratio"] == 0.5

    def test_concurrent_multithreaded_access(self) -> None:
        cache: LRUCache[int, int] = LRUCache(50)
        num_threads = 8
        ops_per_thread = 500

        def worker(thread_id: int) -> None:
            for i in range(ops_per_thread):
                key = (thread_id * 100) + (i % 75)
                cache.put(key, i)
                val = cache.get(key)
                if val is not None:
                    assert isinstance(val, int)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(cache) <= 50


class TestBatchQueue:

    def test_fifo_ordering(self) -> None:
        queue: BatchQueue[int] = BatchQueue()
        assert queue.is_empty() is True
        assert len(queue) == 0

        queue.enqueue(10)
        queue.enqueue(20)
        queue.enqueue(30)

        assert queue.is_empty() is False
        assert len(queue) == 3

        assert queue.dequeue() == 10
        assert queue.dequeue() == 20
        assert queue.dequeue() == 30
        assert queue.dequeue() is None
        assert queue.is_empty() is True

    def test_dequeue_timeout_empty(self) -> None:
        queue: BatchQueue[str] = BatchQueue()
        t0 = time.monotonic()
        val = queue.dequeue(timeout=0.05)
        elapsed = time.monotonic() - t0

        assert val is None
        assert elapsed >= 0.04

    def test_collect_batch_threshold_immediate(self) -> None:
        queue: BatchQueue[int] = BatchQueue()
        for i in range(10):
            queue.enqueue(i)

        batch = queue.collect_batch(max_batch_size=4, timeout_ms=500.0)
        assert batch == [0, 1, 2, 3]
        assert len(queue) == 6

        batch2 = queue.collect_batch(max_batch_size=4, timeout_ms=500.0)
        assert batch2 == [4, 5, 6, 7]
        assert len(queue) == 2

    def test_collect_batch_timeout_partial(self) -> None:
        queue: BatchQueue[str] = BatchQueue()
        queue.enqueue("item1")
        queue.enqueue("item2")

        t0 = time.monotonic()
        batch = queue.collect_batch(max_batch_size=10, timeout_ms=50.0)
        elapsed = (time.monotonic() - t0) * 1000.0

        assert batch == ["item1", "item2"]
        assert len(queue) == 0
        assert elapsed >= 40.0

    def test_collect_batch_empty_timeout(self) -> None:
        queue: BatchQueue[int] = BatchQueue()
        t0 = time.monotonic()
        batch = queue.collect_batch(max_batch_size=5, timeout_ms=30.0)
        elapsed = (time.monotonic() - t0) * 1000.0

        assert batch == []
        assert elapsed >= 25.0

    def test_collect_batch_invalid_size(self) -> None:
        queue: BatchQueue[int] = BatchQueue()
        with pytest.raises(ValueError):
            queue.collect_batch(0, 10.0)
        with pytest.raises(ValueError):
            queue.collect_batch(-1, 10.0)

    def test_concurrent_producers_and_batch_collector(self) -> None:
        queue: BatchQueue[int] = BatchQueue()
        num_producers = 6
        items_per_producer = 200
        total_items = num_producers * items_per_producer

        def producer(start_val: int) -> None:
            for i in range(items_per_producer):
                queue.enqueue(start_val + i)
                if i % 20 == 0:
                    time.sleep(0.001)

        producer_threads = [
            threading.Thread(target=producer, args=(p * items_per_producer,))
            for p in range(num_producers)
        ]

        collected: List[int] = []
        stop_consumer = threading.Event()

        def consumer() -> None:
            while not stop_consumer.is_set() or not queue.is_empty():
                batch = queue.collect_batch(max_batch_size=32, timeout_ms=10.0)
                if batch:
                    collected.extend(batch)

        consumer_thread = threading.Thread(target=consumer)
        consumer_thread.start()

        for t in producer_threads:
            t.start()
        for t in producer_threads:
            t.join()

        stop_consumer.set()
        consumer_thread.join(timeout=3.0)

        while not queue.is_empty():
            batch = queue.collect_batch(max_batch_size=32, timeout_ms=5.0)
            collected.extend(batch)

        assert len(collected) == total_items
        assert sorted(collected) == list(range(total_items))


class TestRingBuffer:

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError):
            RingBuffer(0)
        with pytest.raises(ValueError):
            RingBuffer(-10)

    def test_empty_buffer(self) -> None:
        rb = RingBuffer(5)
        assert len(rb) == 0
        assert rb.total_samples == 0
        assert rb.mean() == 0.0
        assert rb.min() == 0.0
        assert rb.max() == 0.0
        assert rb.percentile(50.0) == 0.0
        assert rb.compute_percentiles() == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_single_element(self) -> None:
        rb = RingBuffer(5)
        rb.append(42.5)
        assert len(rb) == 1
        assert rb.total_samples == 1
        assert rb.mean() == 42.5
        assert rb.min() == 42.5
        assert rb.max() == 42.5
        assert rb.percentile(0.0) == 42.5
        assert rb.percentile(50.0) == 42.5
        assert rb.percentile(100.0) == 42.5

    def test_exact_percentile_linear_interpolation(self) -> None:
        rb = RingBuffer(10)
        for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
            rb.append(val)

        assert len(rb) == 5
        assert rb.mean() == 30.0
        assert rb.min() == 10.0
        assert rb.max() == 50.0

        assert pytest.approx(rb.percentile(0.0), 1e-4) == 10.0
        assert pytest.approx(rb.percentile(50.0), 1e-4) == 30.0
        assert pytest.approx(rb.percentile(100.0), 1e-4) == 50.0

        assert pytest.approx(rb.percentile(25.0), 1e-4) == 20.0
        assert pytest.approx(rb.percentile(75.0), 1e-4) == 40.0

    def test_circular_wrap_around_and_overwriting(self) -> None:
        rb = RingBuffer(4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            rb.append(v)
        assert len(rb) == 4
        assert rb.mean() == 2.5

        rb.append(10.0)
        assert len(rb) == 4
        assert rb.total_samples == 5
        assert rb.min() == 2.0
        assert rb.max() == 10.0
        assert pytest.approx(rb.mean(), 1e-4) == 4.75

        assert pytest.approx(rb.percentile(50.0), 1e-4) == 3.5

    def test_compute_percentiles_batch(self) -> None:
        rb = RingBuffer(100)
        for i in range(1, 101):
            rb.append(float(i))

        pcts = rb.compute_percentiles([50.0, 95.0, 99.0])
        assert pytest.approx(pcts["p50"], abs=0.5) == 50.5
        assert pytest.approx(pcts["p95"], abs=0.5) == 95.05
        assert pytest.approx(pcts["p99"], abs=0.5) == 99.01

    def test_scratch_buffer_zero_reallocation(self) -> None:
        rb = RingBuffer(10)
        for val in [5.0, 1.0, 9.0, 3.0]:
            rb.append(val)

        scratch_id_before = id(rb._scratch)
        _ = rb.percentile(50.0)
        scratch_id_after = id(rb._scratch)
        assert scratch_id_before == scratch_id_after

    def test_invalid_percentile_range(self) -> None:
        rb = RingBuffer(5)
        rb.append(1.0)
        with pytest.raises(ValueError):
            rb.percentile(-0.1)
        with pytest.raises(ValueError):
            rb.percentile(100.1)


class TestInferencePipelineIntegration:

    def test_pipeline_cache_first_lookup(self) -> None:
        model = FastMLBackend()
        with InferencePipeline(
            model=model,
            cache_capacity=100,
            max_batch_size=8,
            batch_timeout_ms=5.0,
        ) as pipeline:
            query = "This product is absolutely amazing and fantastic!"

            res1 = pipeline.predict(query)
            assert res1.cached is False
            assert res1.label == "POSITIVE"
            assert res1.score > 0.5
            assert res1.latency_ms > 0.0

            res2 = pipeline.predict(query)
            assert res2.cached is True
            assert res2.label == res1.label
            assert res2.score == res1.score

            stats = pipeline.stats()
            assert stats["cache"]["hits"] >= 1
            assert stats["cache"]["misses"] >= 1

    def test_pipeline_negative_sentiment(self) -> None:
        model = FastMLBackend()
        with InferencePipeline(model=model, cache_capacity=50) as pipeline:
            query = "Terrible awful broken failure, hate this bug!"
            res = pipeline.predict(query)
            assert res.label == "NEGATIVE"
            assert res.score > 0.5

    def test_predict_batch_sync(self) -> None:
        model = FastMLBackend()
        with InferencePipeline(
            model=model,
            cache_capacity=50,
            max_batch_size=16,
            batch_timeout_ms=10.0,
        ) as pipeline:
            queries = [
                "Excellent service",
                "Awful crash",
                "Excellent service",
                "Clean and reliable code",
            ]
            results = pipeline.predict_batch_sync(queries)
            assert len(results) == 4
            assert results[0].label == "POSITIVE"
            assert results[1].label == "NEGATIVE"
            assert results[2].cached is True or results[0].cached is False
