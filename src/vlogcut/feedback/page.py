"""Static variant comparison pages and loading their downloaded feedback."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import TypeAdapter, ValidationError

from vlogcut.schemas import Feedback


@dataclass(frozen=True)
class VariantEntry:
    """A variant ID and preview href, preserved exactly as supplied."""

    variant_id: str
    preview: str

    def __post_init__(self) -> None:
        if not self.variant_id.strip():
            raise ValueError("variant_id must not be blank")


def build_variants_page(
    variants: Sequence[VariantEntry], *, title: str = "vlogcut variants", context: str = ""
) -> str:
    """Build a deterministic, self-contained page that downloads feedback JSON."""
    if not variants:
        raise ValueError("at least one variant is required")
    seen: set[str] = set()
    cards: list[str] = []
    for index, variant in enumerate(variants):
        key = variant.variant_id.strip()
        if key in seen:
            raise ValueError(f"duplicate variant ID: {key!r}")
        seen.add(key)
        cards.append(f'''<article class="variant" id="variant-{index}">
  <h2>{escape(variant.variant_id, quote=True)}</h2>
  <video controls preload="metadata" src="{escape(variant.preview, quote=True)}"></video>
  <fieldset>
    <legend>Verdict</legend>
    <label><input type="radio" name="verdict-{index}" value="pick"> Pick</label>
    <label><input type="radio" name="verdict-{index}" value="reject"> Reject</label>
    <label><input type="radio" name="verdict-{index}" value="" checked> Unset</label>
  </fieldset>
  <label for="text-{index}">Feedback</label>
  <textarea id="text-{index}" rows="4"></textarea>
</article>''')
    ids = json.dumps([variant.variant_id for variant in variants])
    ids = ids.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    context_html = f'<p id="context">{escape(context, quote=True)}</p>\n' if context else ""
    escaped_title = escape(title, quote=True)
    cards_html = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
.variants {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
.variant {{ flex: 1 1 20rem; min-width: 0; padding: 1rem; border: 1px solid #aaa; }}
video, textarea {{ width: 100%; box-sizing: border-box; }}
h2, #context {{ overflow-wrap: anywhere; white-space: pre-wrap; }}
fieldset {{ margin: 1rem 0; }}
button {{ margin-top: 1rem; padding: 0.75rem; }}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
{context_html}<main class="variants">
{cards_html}
</main>
<button type="button" id="save-feedback">Save feedback</button>
<script>
const variantIds = {ids};
document.getElementById("save-feedback").addEventListener("click", () => {{
  const feedback = [];
  variantIds.forEach((variant_id, index) => {{
    const card = document.getElementById(`variant-${{index}}`);
    const verdict = card.querySelector('input[type="radio"]:checked').value || null;
    const text = card.querySelector("textarea").value;
    if (verdict !== null || text.trim() !== "") {{
      feedback.push({{variant_id, verdict, text}});
    }}
  }});
  const blob = new Blob([JSON.stringify(feedback, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "feedback.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}});
</script>
</body>
</html>
"""


def write_variants_page(
    path: Path,
    variants: Sequence[VariantEntry],
    *,
    title: str = "vlogcut variants",
    context: str = "",
) -> Path:
    """Atomically replace a page using a UTF-8 temporary file beside it."""
    document = build_variants_page(variants, title=title, context=context)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(document)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_feedback(path: Path) -> list[Feedback]:
    """Validate a downloaded JSON array, retaining entry order and supplied strings."""
    try:
        return TypeAdapter(list[Feedback]).validate_json(path.read_bytes())
    except ValidationError as exc:
        detail = " ".join(exc.errors()[0]["msg"].split())
        raise ValueError(f"invalid feedback in {str(path)!r}: {detail}") from exc
