"""Static feedback pages, JSON loading and pure preference-pair derivation."""

from .page import VariantEntry, build_variants_page, load_feedback, write_variants_page
from .pairs import preference_pairs

__all__ = [
    "VariantEntry",
    "build_variants_page",
    "load_feedback",
    "preference_pairs",
    "write_variants_page",
]
