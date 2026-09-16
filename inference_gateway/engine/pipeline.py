from __future__ import annotations
import abc
import concurrent.futures
import hashlib
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dsa.lru_cache import LRUCache
from dsa.batch_queue import BatchQueue
from dsa.ring_buffer import RingBuffer


@dataclass(frozen=True)
class InferenceResult:
    text: str
    label: str
    score: float
    cached: bool
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "score": round(self.score, 4),
            "cached": self.cached,
            "latency_ms": round(self.latency_ms, 4),
        }


class InferenceRequest:
    __slots__ = ("text", "hash_key", "created_at", "future")

    def __init__(self, text: str, hash_key: str) -> None:
        self.text: str = text
        self.hash_key: str = hash_key
        self.created_at: float = time.perf_counter()
        self.future: concurrent.futures.Future[InferenceResult] = concurrent.futures.Future()


class BaseSentimentModel(abc.ABC):

    @abc.abstractmethod
    def infer_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        pass


class FastMLBackend(BaseSentimentModel):

    def __init__(self) -> None:
        self._positive_lexicon: Dict[str, float] = {
            "great": 1.8, "excellent": 2.2, "good": 1.2, "amazing": 2.4, "love": 2.1,
            "fast": 1.4, "best": 2.0, "awesome": 2.3, "happy": 1.5, "positive": 1.6,
            "wonderful": 2.2, "fantastic": 2.3, "reliable": 1.7, "clean": 1.3, "smooth": 1.5,
            "efficient": 1.8, "superior": 1.9, "superb": 2.1, "delightful": 2.0, "perfect": 2.5
        }
        self._negative_lexicon: Dict[str, float] = {
            "bad": -1.8, "terrible": -2.4, "poor": -1.5, "slow": -1.6, "hate": -2.2,
            "worst": -2.5, "awful": -2.3, "bug": -1.4, "fail": -1.9, "negative": -1.5,
            "horrible": -2.4, "broken": -2.0, "waste": -2.1, "annoying": -1.7, "frustrating": -2.0,
            "lag": -1.5, "crash": -2.2, "defect": -1.8, "flaw": -1.6, "useless": -2.3
        }
        self._bias: float = 0.05

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z]+\b", text.lower())

    def _predict_single(self, text: str) -> Tuple[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return "POSITIVE", 0.51

        logit = self._bias
        for token in tokens:
            logit += self._positive_lexicon.get(token, 0.0)
            logit += self._negative_lexicon.get(token, 0.0)

        logit_clamped = max(-20.0, min(20.0, logit))
        prob_positive = 1.0 / (1.0 + math.exp(-logit_clamped))

        if prob_positive >= 0.5:
            return "POSITIVE", prob_positive
        else:
            return "NEGATIVE", 1.0 - prob_positive

    def infer_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [self._predict_single(t) for t in texts]


class TransformerBackend(BaseSentimentModel):

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        from transformers import pipeline
        self._pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1,
            truncation=True,
            max_length=128,
        )

    def infer_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        outputs = self._pipe(texts)
        results: List[Tuple[str, float]] = []
        for out in outputs:
            results.append((out["label"].upper(), float(out["score"])))
        return results


def build_default_backend() -> BaseSentimentModel:
    try:
        import torch
        import transformers
        return TransformerBackend()
    except Exception:
        return FastMLBackend()


class InferencePipeline:

    def __init__(
        self,
        model: Optional[BaseSentimentModel] = None,
        cache_capacity: int = 10_000,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 10.0,
        metrics_buffer_size: int = 50_000,
    ) -> None:
        self.model: BaseSentimentModel = model if model is not None else build_default_backend()
        self.cache: LRUCache[str, Dict[str, Any]] = LRUCache(capacity=cache_capacity)
        self.batch_queue: BatchQueue[InferenceRequest] = BatchQueue()
        self.max_batch_size: int = max_batch_size
        self.batch_timeout_ms: float = batch_timeout_ms

        self.overall_metrics: RingBuffer = RingBuffer(capacity=metrics_buffer_size)
        self.hit_metrics: RingBuffer = RingBuffer(capacity=metrics_buffer_size)
        self.miss_metrics: RingBuffer = RingBuffer(capacity=metrics_buffer_size)

        self._running: bool = False
        self._worker_thread: Optional[threading.Thread] = None
        self._batches_processed: int = 0
        self._total_inferred_items: int = 0
        self._stats_lock: threading.Lock = threading.Lock()

        self.start()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._batch_worker_loop,
            name="BatchInferenceWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.batch_queue.enqueue(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def __enter__(self) -> InferencePipeline:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _batch_worker_loop(self) -> None:
        while self._running:
            batch = self.batch_queue.collect_batch(
                max_batch_size=self.max_batch_size,
                timeout_ms=self.batch_timeout_ms,
            )

            active_batch = [req for req in batch if req is not None]
            if not active_batch:
                continue

            texts = [req.text for req in active_batch]
            try:
                predictions = self.model.infer_batch(texts)
            except Exception as exc:
                for req in active_batch:
                    if not req.future.done():
                        req.future.set_exception(exc)
                continue

            now = time.perf_counter()
            with self._stats_lock:
                self._batches_processed += 1
                self._total_inferred_items += len(active_batch)

            for req, (label, score) in zip(active_batch, predictions):
                latency_ms = (now - req.created_at) * 1000.0

                cache_payload = {"label": label, "score": score}
                self.cache.put(req.hash_key, cache_payload)

                self.miss_metrics.append(latency_ms)
                self.overall_metrics.append(latency_ms)

                result = InferenceResult(
                    text=req.text,
                    label=label,
                    score=score,
                    cached=False,
                    latency_ms=latency_ms,
                )
                if not req.future.done():
                    req.future.set_result(result)

    def predict(self, text: str, timeout_sec: float = 10.0) -> InferenceResult:
        t_start = time.perf_counter()
        hash_key = self.hash_text(text)

        cached_val = self.cache.get(hash_key)
        if cached_val is not None:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.hit_metrics.append(latency_ms)
            self.overall_metrics.append(latency_ms)
            return InferenceResult(
                text=text,
                label=cached_val["label"],
                score=cached_val["score"],
                cached=True,
                latency_ms=latency_ms,
            )

        request = InferenceRequest(text=text, hash_key=hash_key)
        self.batch_queue.enqueue(request)

        try:
            return request.future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Inference request timed out after {timeout_sec}s")

    def predict_batch_sync(self, texts: List[str], timeout_sec: float = 30.0) -> List[InferenceResult]:
        results: List[Optional[InferenceResult]] = [None] * len(texts)
        pending_futures: List[Tuple[int, concurrent.futures.Future[InferenceResult]]] = []

        for idx, text in enumerate(texts):
            t_start = time.perf_counter()
            hash_key = self.hash_text(text)
            cached_val = self.cache.get(hash_key)
            if cached_val is not None:
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                self.hit_metrics.append(latency_ms)
                self.overall_metrics.append(latency_ms)
                results[idx] = InferenceResult(
                    text=text,
                    label=cached_val["label"],
                    score=cached_val["score"],
                    cached=True,
                    latency_ms=latency_ms,
                )
            else:
                req = InferenceRequest(text=text, hash_key=hash_key)
                self.batch_queue.enqueue(req)
                pending_futures.append((idx, req.future))

        for idx, fut in pending_futures:
            results[idx] = fut.result(timeout=timeout_sec)

        return [r for r in results if r is not None]

    def stats(self) -> Dict[str, Any]:
        cache_telemetry = self.cache.stats()
        hit_pct = self.hit_metrics.compute_percentiles([50.0, 95.0, 99.0])
        miss_pct = self.miss_metrics.compute_percentiles([50.0, 95.0, 99.0])
        overall_pct = self.overall_metrics.compute_percentiles([50.0, 95.0, 99.0])

        with self._stats_lock:
            avg_batch_sz = (
                self._total_inferred_items / self._batches_processed
                if self._batches_processed > 0 else 0.0
            )

        return {
            "cache": cache_telemetry,
            "batches_processed": self._batches_processed,
            "total_inferred_items": self._total_inferred_items,
            "avg_batch_size": round(avg_batch_sz, 2),
            "latency_ms": {
                "overall": {
                    "mean": round(self.overall_metrics.mean(), 4),
                    "min": round(self.overall_metrics.min(), 4),
                    "max": round(self.overall_metrics.max(), 4),
                    **overall_pct,
                },
                "cache_hits": {
                    "mean": round(self.hit_metrics.mean(), 4),
                    "min": round(self.hit_metrics.min(), 4),
                    "max": round(self.hit_metrics.max(), 4),
                    **hit_pct,
                },
                "cache_misses": {
                    "mean": round(self.miss_metrics.mean(), 4),
                    "min": round(self.miss_metrics.min(), 4),
                    "max": round(self.miss_metrics.max(), 4),
                    **miss_pct,
                },
            },
        }
