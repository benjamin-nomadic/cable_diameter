import csv
import os

_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "acsr_catalog.csv")


def _load_catalog(path=_CATALOG_PATH):
    catalog = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            catalog.append({
                "code_word": row["code_word"],
                "size": row["size"],
                "stranding": row["stranding"],
                "diameter_mm": float(row["diameter_in"]) * 25.4,
            })
    return catalog


_CATALOG = _load_catalog()


def classify(diameter_mm, tolerance_mm):
    """Return every catalog entry within tolerance_mm of the measured diameter.

    Sorted closest-first. Each entry includes delta_mm (how far its diameter is from
    the measurement), so callers can show the match quality rather than a false-precision
    single answer. Many real ACSR conductors of genuinely different sizes share the same
    (or near-identical) outer diameter, so returning a set rather than one guess is
    intentional, not a fallback.
    """
    candidates = [
        {**entry, "delta_mm": round(abs(entry["diameter_mm"] - diameter_mm), 3)}
        for entry in _CATALOG
        if abs(entry["diameter_mm"] - diameter_mm) <= tolerance_mm
    ]
    candidates.sort(key=lambda c: c["delta_mm"])
    return candidates
