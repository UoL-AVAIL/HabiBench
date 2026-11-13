"""Offline visual review of a local LUCAS context-screening run."""
import base64
import io
import os
from html import escape
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageOps

from .pipeline import atomic_text


GROUPS = {
    "excluded": "Excluded",
    "ambiguous": "Kept with uncertain context",
    "kept": "Kept",
    "error": "Review errors",
}
VIEWS = {"P": "Point", "N": "North", "E": "East", "S": "South", "W": "West", "C": "Close-up"}
STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f6f6f1;color:#162821;font:16px/1.45 'Helvetica Neue',Arial,sans-serif}
main{max-width:1440px;margin:auto;padding:clamp(18px,4vw,60px)}h1,h2,h3,p{margin:0}h1{font:clamp(34px,5vw,58px)/1.05 Georgia,serif}
header p{margin:1rem 0}nav{display:flex;flex-wrap:wrap;gap:.5rem}nav a{border:1px solid #acbcb0;padding:.4rem .75rem;border-radius:.5rem}
a{color:#16523b;text-underline-offset:.18em}section{margin:2.5rem 0}h2{font:30px Georgia,serif;margin-bottom:.8rem}
h2 span,h3 span{font:normal .95rem Arial,sans-serif;color:#53615a}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,315px),1fr));gap:1rem}
article{background:white;border:1px solid #cdd6cf;border-radius:8px;overflow:hidden}img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#e6e9e2}
.body{padding:1rem}h3{font-size:18px}dl{margin:.6rem 0;font-size:.92rem}dl div{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}dt{color:#495b51}dd{margin:0}
blockquote{margin:.8rem 0;border-left:2px solid #7ba68d;padding-left:.7rem}.note{margin:.5rem 0;font-size:.92rem}
"""
MARKER = '<meta name="habibench-gallery" content="context-review">'


def _group(record):
    if record["decision"] == "error":
        return "error"
    if record["decision"] == "exclude":
        return "excluded"
    if record["decision"] != "keep":
        raise ValueError(f"Expected a context-screening review, found {record['decision']!r}")
    return "kept" if record["observation"]["context"] == "compatible" else "ambiguous"


def _thumb(path, alt):
    try:
        with Image.open(path) as src:
            preview = ImageOps.exif_transpose(src).convert("RGB")
        preview.thumbnail((576, 432), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=83)
    except (OSError, ValueError):
        return "<p>Photo preview unavailable.</p>"
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img src="data:image/jpeg;base64,{payload}" loading="lazy" alt="{alt}">'


def _card(root, run, row, record):
    path = (root / row["image_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path outside dataset: {row['image_path']}")
    url = escape(quote(os.path.relpath(path, run), safe="/"), quote=True)
    photo = escape(row["photo_id"])
    thumbnail = _thumb(path, photo)
    original = f'<a href="{url}" target="_blank" rel="noopener">Open original photo</a>' if path.is_file() else ""
    fields = [("Curation decision", record["decision"].capitalize()),
              ("Survey label", f"{row['habitat']} ({row['broad_class']})"), ("View", VIEWS[row["view"]])]
    if record["decision"] == "error":
        reason = record["error"]
    else:
        obs = record["observation"]
        fields.extend([("Qwen context", obs["context"].replace("_", " ")), ("Dominant obstruction", obs["obstruction"].replace("_", " ")),
                       ("Landscape visible", "Yes" if obs["landscape_visible"] else "No"), ("Evidence strength", obs["strength"])])
        reason = obs["reason"]
    facts = "".join(f"<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>" for k,v in fields)
    note = ('<p class="note">Visually retained. The paired training export omits this photo because its Tessera vector is unavailable.</p>'
            if record["decision"] == "keep" and row["vector_status"] != "valid" else "")
    if record["decision"] == "keep" and record["observation"]["context"] == "extreme_outlier":
        note = '<p class="note">Kept under the conservative exclusion rule. ' + (
            'Qwen reported no dominant obstruction.</p>' if record["observation"]["obstruction"] == "none"
            else 'Qwen reported usable landscape context or insufficient evidence.</p>') + note
    return (f'<article id="photo-{photo}">{thumbnail}<div class="body"><h3>{photo}</h3><dl>{facts}</dl>'
            f'<blockquote>{escape(reason)}</blockquote>{note}{original}</div></article>')


def write_gallery(root, run, rows, records, identity):
    """Generate a single HTML summary with embedded previews and relative links to originals."""
    root, run = Path(root).resolve(), Path(run).resolve()
    target = run / "photo-review.html"
    if target.exists() and MARKER not in target.read_text(encoding="utf-8")[:500]:
        raise ValueError("Existing photo-review.html belongs to an earlier workflow. Use a new run directory.")
    groups = {key: [] for key in GROUPS}
    for row in rows:
        rec = records[row["photo_id"]]
        groups[_group(rec)].append(_card(root, run, row, rec))
    navigation, sections = [], []
    for key, title in GROUPS.items():
        cards = groups[key]
        if cards:
            navigation.append(f'<a href="#{key}">{title} ({len(cards)})</a>')
            sections.append(f'<section id="{key}"><h2>{title} <span>{len(cards)} photos</span></h2>'
                            f'<div class="grid">{"".join(cards)}</div></section>')
    text = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">{MARKER}'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>LUCAS context review</title>'
            f'<style>{STYLE}</style></head><body><main><header><h1>LUCAS context review</h1>'
            f'<p>{len(rows)} photos · {escape(identity["model"])}. Qwen received the broad survey label for each photo.</p>'
            f'<nav aria-label="Curation decisions">{"".join(navigation)}</nav></header>{"".join(sections)}</main></body></html>')
    atomic_text(target, text)
    return target
