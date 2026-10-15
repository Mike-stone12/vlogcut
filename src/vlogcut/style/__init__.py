"""Style: reference videos -> explicit, editable `StyleProfile`; distances between profiles."""

from vlogcut.style.distance import StyleDistance, profile_distance, profile_from_edl
from vlogcut.style.profile import (
    ReferenceParams,
    StyleError,
    analyze_reference,
    build_profile,
    load_profile,
)

__all__ = [
    "ReferenceParams",
    "StyleDistance",
    "StyleError",
    "analyze_reference",
    "build_profile",
    "load_profile",
    "profile_distance",
    "profile_from_edl",
]
