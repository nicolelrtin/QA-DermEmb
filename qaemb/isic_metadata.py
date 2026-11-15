"""Fetch the public ISIC Archive metadata for every image in a manifest (PROJECT.md, Plan item 2).

  python -m qaemb.isic_metadata data/train_10class_100each/manifest.csv

Writes isic_metadata.csv next to the manifest, one row per image: image type (dermoscopic / clinical), contributing
institution and licence, patient and lesion IDs, approximate age, sex, body site (anatom_site_1, coarse; anatom_site_2, finer), lesion size, the ISIC diagnosis hierarchy
(diagnosis_1 to diagnosis_5), how the diagnosis was confirmed, and image size. Fields ISIC does not have for an image are
left empty. The raw JSON of each image is kept in isic_metadata_raw.jsonl, so new fields can be read without
fetching again. Images already in the raw file are not fetched again.

Only the ISIC image IDs are sent (public identifiers), with a generic client name.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://api.isic-archive.com/api/v2/images/{}/"
HEADERS = {"User-Agent": "qaemb-research-client", "Accept": "application/json"}


def fetch(isic_id: str, tries: int = 4) -> dict:
    for k in range(tries):
        try:
            r = requests.get(API.format(isic_id), headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {"isic_id": isic_id, "_missing": True}
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise SystemExit(f"Could not fetch {isic_id} after {tries} tries")


def flatten(j: dict) -> dict:
    md = j.get("metadata", {})
    acq, cli = md.get("acquisition", {}), md.get("clinical", {})
    row = {"isic_id": j["isic_id"], "missing": bool(j.get("_missing")),
           "attribution": j.get("attribution"), "copyright_license": j.get("copyright_license")}
    for k in ("image_type", "image_manipulation", "dermoscopic_type", "pixels_x", "pixels_y"):
        row[k] = acq.get(k)
    for k in ("patient_id", "lesion_id", "age_approx", "sex", "anatom_site_1", "anatom_site_2", "anatom_site_special",
              "clin_size_long_diam_mm", "concomitant_biopsy", "tbp_tile_type",
              "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5", "diagnosis_confirm_type",
              "melanocytic", "fitzpatrick_skin_type", "family_hx_mm", "personal_hx_mm", "mel_thick_mm"):
        row[k] = cli.get(k)
    return row


def main(manifest: str) -> None:
    man = Path(manifest)
    ids = pd.read_csv(man, dtype=str).isic_id.tolist()
    raw_path = man.parent / "isic_metadata_raw.jsonl"
    have = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            j = json.loads(line)
            have[j["isic_id"]] = j
    todo = [i for i in ids if i not in have]
    print(f"{len(ids)} images, {len(have)} already fetched, {len(todo)} to fetch")
    with ThreadPoolExecutor(6) as ex, open(raw_path, "a") as fh:
        for j in ex.map(fetch, todo):
            have[j["isic_id"]] = j
            fh.write(json.dumps(j) + "\n")
    out = pd.DataFrame([flatten(have[i]) for i in ids])
    out.to_csv(man.parent / "isic_metadata.csv", index=False)
    print(f"wrote {man.parent / 'isic_metadata.csv'}; missing from ISIC: {int(out.missing.sum())}")
    for c in ("image_type", "attribution", "sex", "anatom_site_1", "age_approx", "patient_id", "lesion_id"):
        print(f"  {c}: {out[c].notna().sum()} of {len(out)} known, {out[c].nunique()} distinct")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/train_10class_100each/manifest.csv")
