import json
import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import pytest

from vlogcut.feedback import (
    VariantEntry,
    build_variants_page,
    load_feedback,
    preference_pairs,
    write_variants_page,
)
from vlogcut.schemas import Feedback, PreferencePair


class Elements(HTMLParser):
    def __init__(self, document: str) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.feed(document)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))

    def attributes(self, tag: str) -> list[dict[str, str | None]]:
        return [attrs for name, attrs in self.tags if name == tag]


def test_page_structure_order_controls_and_determinism() -> None:
    variants = [
        VariantEntry(" second ", "preview second.mp4"),
        VariantEntry("first", 'folder/first.mp4?a=1&b="two"'),
    ]
    page = build_variants_page(variants)
    assert page.startswith("<!DOCTYPE html>")
    assert page.index("<h2> second </h2>") < page.index("<h2>first</h2>")
    assert page == build_variants_page(variants)
    assert "http://" not in page and "https://" not in page
    assert 'id="context"' not in page
    elements = Elements(page)
    assert len(elements.attributes("article")) == 2
    videos = elements.attributes("video")
    assert [video["src"] for video in videos] == [entry.preview for entry in variants]
    assert all("controls" in video and video["preload"] == "metadata" for video in videos)
    inputs = elements.attributes("input")
    for index in range(2):
        group = [item for item in inputs if item["name"] == f"verdict-{index}"]
        assert [item["value"] for item in group] == ["pick", "reject", ""]
        assert all(item["type"] == "radio" for item in group)
        assert [item["value"] for item in group if "checked" in item] == [""]
    assert len(elements.attributes("textarea")) == 2
    assert len(elements.attributes("button")) == 1
    assert "Save feedback</button>" in page
    # Inspect the download contract without running JavaScript.
    assert "variantIds.forEach((variant_id, index)" in page
    assert ".value || null" in page
    assert 'verdict !== null || text.trim() !== ""' in page
    assert "feedback.push({variant_id, verdict, text})" in page
    assert "new Blob([JSON.stringify(feedback, null, 2)]" in page
    assert "URL.createObjectURL(blob)" in page
    assert 'link.download = "feedback.json"' in page
    assert "link.click()" in page
    assert "URL.revokeObjectURL(url)" in page


def test_escaping_html_and_script_data() -> None:
    dangerous = "<b>&\"'</script><script>alert(1)</script>"
    variants = [VariantEntry(dangerous, dangerous)]
    page = build_variants_page(variants, title=dangerous, context=dangerous)
    escaped = escape(dangerous, quote=True)
    assert dangerous not in page
    assert f"<title>{escaped}</title>" in page
    assert f"<h1>{escaped}</h1>" in page
    assert f"<h2>{escaped}</h2>" in page
    assert f'<p id="context">{escaped}</p>' in page
    assert f'src="{escaped}"' in page
    assert page.count("<script>") == page.count("</script>") == 1
    match = re.search(r"const variantIds = (.*);", page)
    assert match is not None
    encoded = match.group(1)
    assert all(character not in encoded for character in "<>&")
    assert all(escape_code in encoded for escape_code in (r"\u003c", r"\u003e", r"\u0026"))
    assert json.loads(encoded) == [dangerous]


@pytest.mark.parametrize("variant_id", ["", " \n\t"])
def test_blank_variant_id(variant_id: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        VariantEntry(variant_id, "preview.mp4")


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["a", " a "], ["a\nb", " a\nb "]])
def test_invalid_variant_lists(ids: list[str]) -> None:
    with pytest.raises(ValueError) as exc:
        build_variants_page([VariantEntry(value, "preview.mp4") for value in ids])
    assert len(str(exc.value).splitlines()) == 1


def test_write_page_replaces_existing_file_without_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "variants.html"
    path.write_text("old page", encoding="utf-8")
    variants = [VariantEntry("café", "café.mp4")]
    assert write_variants_page(path, variants, title="日常", context="A day") == path
    expected = build_variants_page(variants, title="日常", context="A day")
    assert path.read_bytes() == expected.encode("utf-8")
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "variants.html"
    path.write_text("old page", encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        assert source.parent == target.parent == tmp_path
        assert source.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_variants_page(path, [VariantEntry("a", "a.mp4")])
    assert path.read_text(encoding="utf-8") == "old page"
    assert list(tmp_path.iterdir()) == [path]


def test_load_round_trip_and_preference_pairs(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    path.write_text(
        '[{"variant_id":"r","verdict":"reject","text":""},'
        '{"variant_id":" note ","verdict":null,"text":" café \\n"},'
        '{"variant_id":" p ","verdict":"pick","text":" good "},'
        '{"variant_id":"p2","verdict":"pick","text":""}]',
        encoding="utf-8",
    )
    feedback = load_feedback(path)
    assert all(isinstance(entry, Feedback) for entry in feedback)
    assert [entry.model_dump() for entry in feedback] == json.loads(
        path.read_text(encoding="utf-8")
    )
    assert preference_pairs(feedback, context="A day") == [
        PreferencePair(context="A day", chosen=" p ", rejected="r"),
        PreferencePair(context="A day", chosen="p2", rejected="r"),
    ]
    path.write_text("[]", encoding="utf-8")
    assert load_feedback(path) == []


@pytest.mark.parametrize(
    "payload",
    [
        "{broken",
        '{"variant_id":"a","verdict":"pick","text":""}',
        '[{"variant_id":" ","verdict":"pick","text":""}]',
        '[{"variant_id":"a","verdict":null,"text":" \\n"}]',
        '[{"variant_id":"a","verdict":"maybe","text":""}]',
    ],
)
def test_load_errors_name_file_on_one_line(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "feedback.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_feedback(path)
    assert path.name in str(exc.value)
    assert len(str(exc.value).splitlines()) == 1
