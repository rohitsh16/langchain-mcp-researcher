"""
Literature subsystem for multi-index search, ranking, and deduplication.
"""
from research.literature.search import (
    PaperIndex,
    ArxivIndex,
    OpenAlexIndex,
    MockIndex,
    MultiSourceSearchPlanner,
)
from research.literature.dedup import PaperDeduplicator

__all__ = [
    "PaperIndex",
    "ArxivIndex",
    "OpenAlexIndex",
    "MockIndex",
    "MultiSourceSearchPlanner",
    "PaperDeduplicator",
]
