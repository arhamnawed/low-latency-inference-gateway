from __future__ import annotations
import time
import threading
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class QueueNode(Generic[T]):
    __slots__ = ("value", "next")

    def __init__(self, value: T) -> None:
        self.value: T = value
        self.next: Optional[QueueNode[T]] = None


class BatchQueue(Generic[T]):

    def __init__(self) -> None:
        self._head: Optional[QueueNode[T]] = None
        self._tail: Optional[QueueNode[T]] = None
        self._size: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._not_empty: threading.Condition = threading.Condition(self._lock)

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def is_empty(self) -> bool:
        with self._lock:
            return self._size == 0

    def enqueue(self, item: T) -> None:
        new_node = QueueNode(item)
        with self._not_empty:
            if self._tail is None:
                self._head = new_node
                self._tail = new_node
            else:
                self._tail.next = new_node
                self._tail = new_node
            self._size += 1
            self._not_empty.notify()

    def dequeue(self, timeout: Optional[float] = None) -> Optional[T]:
        with self._not_empty:
            if self._size == 0:
                if timeout is None or timeout <= 0:
                    return None
                if not self._not_empty.wait(timeout=timeout):
                    return None
                if self._size == 0:
                    return None

            node = self._head
            assert node is not None
            self._head = node.next
            if self._head is None:
                self._tail = None
            self._size -= 1
            node.next = None
            return node.value

    def collect_batch(self, max_batch_size: int, timeout_ms: float) -> List[T]:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")

        batch: List[T] = []
        timeout_sec = max(0.0, timeout_ms / 1000.0)
        deadline = time.monotonic() + timeout_sec

        with self._not_empty:
            while len(batch) < max_batch_size:
                while self._head is not None and len(batch) < max_batch_size:
                    node = self._head
                    self._head = node.next
                    if self._head is None:
                        self._tail = None
                    self._size -= 1
                    node.next = None
                    batch.append(node.value)

                if len(batch) >= max_batch_size:
                    break

                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    break

                self._not_empty.wait(timeout=remaining_time)

        return batch

    def clear(self) -> None:
        with self._lock:
            curr = self._head
            while curr is not None:
                next_node = curr.next
                curr.next = None
                curr = next_node
            self._head = None
            self._tail = None
            self._size = 0
