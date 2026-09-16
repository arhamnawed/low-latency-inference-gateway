from __future__ import annotations
import threading
from typing import Generic, Hashable, List, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class Node(Generic[K, V]):
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Optional[K] = None, value: Optional[V] = None) -> None:
        self.key: Optional[K] = key
        self.value: Optional[V] = value
        self.prev: Optional[Node[K, V]] = None
        self.next: Optional[Node[K, V]] = None

    def __repr__(self) -> str:
        return f"Node(key={self.key!r}, value={self.value!r})"


class DoublyLinkedList(Generic[K, V]):

    def __init__(self) -> None:
        self._head: Node[K, V] = Node()
        self._tail: Node[K, V] = Node()
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def add_first(self, node: Node[K, V]) -> None:
        node.prev = self._head
        node.next = self._head.next
        assert self._head.next is not None
        self._head.next.prev = node
        self._head.next = node
        self._size += 1

    def remove(self, node: Node[K, V]) -> Node[K, V]:
        if node.prev is None or node.next is None:
            raise ValueError("Node is not linked in DoublyLinkedList")

        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

        node.prev = None
        node.next = None
        self._size -= 1
        return node

    def move_to_front(self, node: Node[K, V]) -> None:
        self.remove(node)
        self.add_first(node)

    def pop_tail(self) -> Optional[Node[K, V]]:
        if self._size == 0:
            return None
        lru_node = self._tail.prev
        assert lru_node is not None and lru_node is not self._head
        return self.remove(lru_node)

    def to_list(self) -> List[Tuple[K, V]]:
        items: List[Tuple[K, V]] = []
        curr = self._head.next
        while curr is not None and curr is not self._tail:
            if curr.key is not None:
                items.append((curr.key, curr.value))  # type: ignore[arg-type]
            curr = curr.next
        return items


class LRUCache(Generic[K, V]):

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Cache capacity must be a positive integer, got {capacity}")
        self._capacity: int = capacity
        self._map: dict[K, Node[K, V]] = {}
        self._list: DoublyLinkedList[K, V] = DoublyLinkedList()
        self._lock: threading.RLock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            node = self._map.get(key)
            if node is None:
                self._misses += 1
                return default

            self._list.move_to_front(node)
            self._hits += 1
            return node.value

    def put(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._map:
                node = self._map[key]
                node.value = value
                self._list.move_to_front(node)
                return

            if len(self._map) >= self._capacity:
                lru_node = self._list.pop_tail()
                if lru_node is not None and lru_node.key is not None:
                    del self._map[lru_node.key]
                    self._evictions += 1

            new_node = Node(key=key, value=value)
            self._list.add_first(new_node)
            self._map[key] = new_node

    def peek(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            node = self._map.get(key)
            if node is None:
                return default
            return node.value

    def contains(self, key: K) -> bool:
        with self._lock:
            return key in self._map

    def get_keys_ordered(self) -> List[K]:
        with self._lock:
            return [k for k, _ in self._list.to_list()]

    def get_items_ordered(self) -> List[Tuple[K, V]]:
        with self._lock:
            return self._list.to_list()

    def clear(self) -> None:
        with self._lock:
            self._map.clear()
            self._list = DoublyLinkedList()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total_lookups = self._hits + self._misses
            hit_ratio = (self._hits / total_lookups) if total_lookups > 0 else 0.0
            return {
                "capacity": self._capacity,
                "size": len(self._map),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_ratio": round(hit_ratio, 4),
            }
