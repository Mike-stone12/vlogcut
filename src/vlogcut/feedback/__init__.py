"""Feedback pages, preference pairs, classification and pure profile updates."""

from .classify import (
    FeedbackClassification,
    FeedbackClassificationError,
    ProfileDelta,
    apply_profile_delta,
    build_feedback_messages,
    classify_feedback,
    parse_classification,
)
from .page import VariantEntry, build_variants_page, load_feedback, write_variants_page
from .pairs import preference_pairs

__all__ = [
    "FeedbackClassification",
    "FeedbackClassificationError",
    "ProfileDelta",
    "VariantEntry",
    "apply_profile_delta",
    "build_feedback_messages",
    "build_variants_page",
    "classify_feedback",
    "load_feedback",
    "parse_classification",
    "preference_pairs",
    "write_variants_page",
]
