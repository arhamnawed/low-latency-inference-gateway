from __future__ import annotations
import math
import threading
from typing import Dict, List, Optional


class RingBuffer:

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be greater than 0, got {capacity}")
        self._capacity: int = capacity
        self._buffer: List[float] = [0.0] * capacity
        self._scratch: List[float] = [0.0] * capacity
        self._write_index: int = 0
        self._count: int = 0
        self._rolling_sum: float = 0.0
        self._lock: threading.Lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return min(self._count, self._capacity)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._count

    def append(self, value: float) -> None:
        with self._lock:
            if self._count >= self._capacity:
                old_val = self._buffer[self._write_index]
                self._rolling_sum -= old_val

            self._buffer[self._write_index] = value
            self._rolling_sum += value
            self._write_index = (self._write_index + 1) % self._capacity
            self._count += 1

    def mean(self) -> float:
        with self._lock:
            k = min(self._count, self._capacity)
            if k == 0:
                return 0.0
            return self._rolling_sum / k

    def min(self) -> float:
        with self._lock:
            k = min(self._count, self._capacity)
            if k == 0:
                return 0.0
            min_val = self._buffer[0]
            for i in range(1, k):
                if self._buffer[i] < min_val:
                    min_val = self._buffer[i]
            return min_val

    def max(self) -> float:
        with self._lock:
            k = min(self._count, self._capacity)
            if k == 0:
                return 0.0
            max_val = self._buffer[0]
            for i in range(1, k):
                if self._buffer[i] > max_val:
                    max_val = self._buffer[i]
            return max_val

    def _quicksort_scratch(self, low: int, high: int) -> None:
        if low < high:
            pivot = self._scratch[high]
            i = low - 1
            for j in range(low, high):
                if self._scratch[j] <= pivot:
                    i += 1
                    self._scratch[i], self._scratch[j] = self._scratch[j], self._scratch[i]
            self._scratch[i + 1], self._scratch[high] = self._scratch[high], self._scratch[i + 1]
            pi = i + 1

            self._quicksort_scratch(low, pi - 1)
            self._quicksort_scratch(pi + 1, high)

    def _copy_to_scratch(self, k: int) -> None:
        for i in range(k):
            self._scratch[i] = self._buffer[i]

    def percentile(self, p: float) -> float:
        if not (0.0 <= p <= 100.0):
            raise ValueError(f"Percentile must be between 0 and 100, got {p}")

        with self._lock:
            k = min(self._count, self._capacity)
            if k == 0:
                return 0.0
            if k == 1:
                return self._buffer[0]

            self._copy_to_scratch(k)
            self._quicksort_scratch(0, k - 1)

            rank = (p / 100.0) * (k - 1)
            low_idx = int(math.floor(rank))
            high_idx = int(math.ceil(rank))
            weight = rank - low_idx

            return (1.0 - weight) * self._scratch[low_idx] + weight * self._scratch[high_idx]

    def compute_percentiles(self, percentiles: Optional[List[float]] = None) -> Dict[str, float]:
        if percentiles is None:
            percentiles = [50.0, 95.0, 99.0]

        with self._lock:
            k = min(self._count, self._capacity)
            res: Dict[str, float] = {}

            if k == 0:
                for p in percentiles:
                    res[f"p{int(p) if p.is_integer() else p}"] = 0.0
                return res

            if k == 1:
                val = self._buffer[0]
                for p in percentiles:
                    res[f"p{int(p) if p.is_integer() else p}"] = val
                return res

            self._copy_to_scratch(k)
            self._quicksort_scratch(0, k - 1)

            for p in percentiles:
                rank = (p / 100.0) * (k - 1)
                low_idx = int(math.floor(rank))
                high_idx = int(math.ceil(rank))
                weight = rank - low_idx
                interpolated = (1.0 - weight) * self._scratch[low_idx] + weight * self._scratch[high_idx]
                key_name = f"p{int(p) if p.is_integer() else p}"
                res[key_name] = round(interpolated, 4)

            return res

    def clear(self) -> None:
        with self._lock:
            self._write_index = 0
            self._count = 0
            self._rolling_sum = 0.0
            for i in range(self._capacity):
                self._buffer[i] = 0.0
                self._scratch[i] = 0.0
