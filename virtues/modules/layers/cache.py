from collections import OrderedDict
from typing import Any


class LRUCache(OrderedDict):
    """
    A simple LRU (Least Recently Used) cache implementation using OrderedDict.
    """

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0, "cache_len must be a positive integer."
        self.cache_len = cache_len
        super().__init__(*args, **kwargs)
        self.cache_hits = 0
        self.cache_misses = 0

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)

        while len(self) > self.cache_len:
            self.popitem(last=False)

    def __getitem__(self, key: Any) -> Any:
        if key not in self:
            self.cache_misses += 1
            return None
        value = super().__getitem__(key)
        self.move_to_end(key)
        self.cache_hits += 1
        return value