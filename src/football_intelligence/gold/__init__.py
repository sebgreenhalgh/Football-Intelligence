"""Canonical, append-only human Gold Corpus (not detector predictions)."""

from .corpus import GoldCorpus, GoldError, discover_root

__all__ = ["GoldCorpus", "GoldError", "discover_root"]
