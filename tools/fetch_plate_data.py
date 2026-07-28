"""Download freely-licensed Egyptian licence-plate images for training/validation.

Source is Wikimedia Commons, chosen because it needs no API key and its files
carry explicit licences — which matters here, since plate images are personal
data and a scraped set of unknown provenance is not something to train on and
ship. Each image's licence and author are recorded in credits.csv.

    python -m tools.fetch_plate_data                  # default: Egypt category
    python -m tools.fetch_plate_data --limit 200

Output under data/plates/egypt_commons/:
    images/<name>.jpg      the images
    credits.csv            file, licence, author, source URL

This is a few dozen images: enough to FIT and CHECK the plate-colour rule, which
is a low-dimensional decision over measured HSV features. It is not enough to
train a detector or an OCR model from scratch — for those, see docs/anpr-plan.md
(EALPR for characters; the pretrained Egyptian ALPR models need no training).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "plates" / "egypt_commons"
API = "https://commons.wikimedia.org/w/api.php"
# Commons asks for a descriptive User-Agent and returns 403 without one.
UA = {"User-Agent": "ITS-Traffic-Research/1.0 (academic traffic analytics)"}

DEFAULT_CATEGORIES = [
    "Category:License plates of Egypt",
]


def _api(params: dict) -> dict:
    params = {**params, "format": "json"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers=UA)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def list_category(cat: str, limit: int) -> list[str]:
    titles, cont = [], None
    while len(titles) < limit:
        p = {"action": "query", "list": "categorymembers", "cmtitle": cat,
             "cmlimit": min(500, limit), "cmtype": "file"}
        if cont:
            p["cmcontinue"] = cont
        d = _api(p)
        titles += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return titles[:limit]


def image_info(titles: list[str], thumb_px: int) -> dict[str, dict]:
    """URL + licence + author for each file, in batches of 50 (the API cap).

    Requests a THUMBNAIL rather than the original. Commons returns 429 for bulk
    downloads of full-resolution originals and asks callers to use thumbnails
    instead; a 1024 px wide plate photo is also far more resolution than the
    colour rule needs, so this costs nothing and is the polite thing to do.
    """
    info: dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        d = _api({"action": "query", "titles": "|".join(titles[i:i + 50]),
                  "prop": "imageinfo",
                  "iiprop": "url|extmetadata", "iiurlwidth": thumb_px})
        for page in d.get("query", {}).get("pages", {}).values():
            ii = (page.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata", {})
            if not ii.get("url"):
                continue
            info[page["title"]] = {
                # thumburl is absent for non-raster files; fall back to the original
                "url": ii.get("thumburl") or ii["url"],
                "descriptionurl": ii.get("descriptionurl", ""),
                "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
                "author": re.sub(r"<[^>]+>", "",
                                 meta.get("Artist", {}).get("value", "")).strip(),
            }
    return info


def safe_name(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title.replace("File:", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--thumb-px", type=int, default=1024,
                    help="request this width instead of the original")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between downloads (be polite to Commons)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    titles: list[str] = []
    for c in args.categories:
        found = list_category(c, args.limit)
        print(f"{c}: {len(found)} files")
        titles += found
    titles = sorted(set(titles))
    if not titles:
        print("nothing found")
        return 1

    info = image_info(titles, args.thumb_px)
    rows = []
    for n, t in enumerate(titles):
        meta = info.get(t)
        if not meta:
            continue
        name = safe_name(t)
        dest = out / "images" / name
        if not dest.exists():
            # Serialised with a pause. Commons rate-limits bulk downloads, and
            # being throttled off mid-run leaves a half-set that silently biases
            # whatever is fitted on it.
            for attempt in range(4):
                try:
                    req = urllib.request.Request(meta["url"], headers=UA)
                    dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                    break
                except Exception as exc:
                    if attempt == 3:
                        print(f"  !! {name}: {exc}")
                    else:
                        time.sleep(2 ** attempt)
            else:
                continue
            time.sleep(args.delay)
        if not dest.exists():
            continue
        if n % 10 == 0:
            print(f"  {n}/{len(titles)} ...", flush=True)
        rows.append({"file": name, "license": meta["license"],
                     "author": meta["author"], "source": meta["descriptionurl"]})

    with open(out / "credits.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "license", "author", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} images -> {out / 'images'}")
    print(f"credits -> {out / 'credits.csv'}")
    from collections import Counter
    print("licences:", dict(Counter(r["license"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
