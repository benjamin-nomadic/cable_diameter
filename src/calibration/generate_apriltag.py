import numpy as np
import cv2

# ── Configure ────────────────────────────────────────────────────────────────
APRILTAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11   # must match compute_homography.py
TAG_ID = 0
TAG_SIZE_MM = 100.0       # target size of the printed tag, edge to edge
DPI = 300
OUTPUT = "apriltag_36h11_id0.png"
# ─────────────────────────────────────────────────────────────────────────────

# A4 landscape: 297 × 210 mm
PAGE_W_MM, PAGE_H_MM = 297, 210

px_per_mm = DPI / 25.4
page_w_px = round(PAGE_W_MM * px_per_mm)
page_h_px = round(PAGE_H_MM * px_per_mm)
tag_px = round(TAG_SIZE_MM * px_per_mm)

if tag_px > min(page_w_px, page_h_px):
    raise ValueError(
        f"Tag ({tag_px}px) does not fit on A4 landscape ({page_w_px}×{page_h_px}px). "
        f"Reduce TAG_SIZE_MM."
    )

aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILY)
tag_img = cv2.aruco.generateImageMarker(aruco_dict, TAG_ID, tag_px)

# Center the tag on a blank page — the surrounding white space is the "quiet zone"
# detection needs around the tag.
page = np.ones((page_h_px, page_w_px), dtype=np.uint8) * 255
off_x = (page_w_px - tag_px) // 2
off_y = (page_h_px - tag_px) // 2
page[off_y:off_y + tag_px, off_x:off_x + tag_px] = tag_img

cv2.imwrite(OUTPUT, page)

print(f"Generated: {OUTPUT}")
print(f"  Family:   tag36h11, ID {TAG_ID}")
print(f"  Tag size: {TAG_SIZE_MM}mm  ({tag_px}px at {DPI} DPI)")
print(f"  Image:    {page_w_px}×{page_h_px}px  (A4 landscape at {DPI} DPI)")
print()
print("PRINTING CHECKLIST:")
print("  1. Print in LANDSCAPE orientation")
print("  2. Set scale to 100% / actual size  (NOT fit-to-page)")
print("  3. Use matte paper — glossy causes reflections that break detection")
print("  4. Mount flat on cardboard or a clipboard — any warp hurts accuracy")
print("  5. After printing, measure the printed tag edge-to-edge (the full")
print("     black-bordered square, not the white paper around it) with a ruler")
print("     and update TAG_SIZE_MM in src/calibration/compute_homography.py")
