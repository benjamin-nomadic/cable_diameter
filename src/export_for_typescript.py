import csv
import json
import os

import numpy as np

CALIBRATION_PATH = "data/calibration.npz"
HOMOGRAPHY_PATH = "data/homography.npz"
CATALOG_PATH = "data/acsr_catalog.csv"

OUTPUT_DIR = os.path.join(
    "..", "nomadic-app", "nomadic_app", "supabase", "functions", "_shared", "cableDiameter"
)


def export_calibration():
    cal = np.load(CALIBRATION_PATH)
    return {"K": cal["K"].tolist(), "dist": cal["dist"].tolist()}


def export_homography():
    hom = np.load(HOMOGRAPHY_PATH)
    return {
        "H": hom["H"].tolist(),
        "output_w": int(hom["output_w"]),
        "output_h": int(hom["output_h"]),
        "pixels_per_mm": float(hom["pixels_per_mm"]),
    }


def export_catalog():
    with open(CATALOG_PATH, newline="") as f:
        return [
            {
                "code_word": row["code_word"],
                "size": row["size"],
                "stranding": row["stranding"],
                "diameter_in": float(row["diameter_in"]),
            }
            for row in csv.DictReader(f)
        ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exports = {
        "calibration.json": export_calibration(),
        "homography.json": export_homography(),
        "acsrCatalog.json": export_catalog(),
    }
    for filename, data in exports.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
