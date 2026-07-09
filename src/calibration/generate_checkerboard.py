import numpy as np
import cv2

# ── Configure ────────────────────────────────────────────────────────────────
INNER_CORNERS = (5, 3)    # (cols, rows) of inner corners  →  10×7 squares
SQUARE_SIZE_MM = 25.0     # target square size in mm
DPI = 300
OUTPUT = "checkerboard_a4_landscape.png"
# ─────────────────────────────────────────────────────────────────────────────

# A4 landscape: 297 × 210 mm
PAGE_W_MM, PAGE_H_MM = 297, 210

px_per_mm = DPI / 25.4
page_w_px = round(PAGE_W_MM * px_per_mm)   # 3508
page_h_px = round(PAGE_H_MM * px_per_mm)   # 2480

num_x = INNER_CORNERS[0] + 1   # number of squares horizontally
num_y = INNER_CORNERS[1] + 1   # number of squares vertically
sq_px = round(SQUARE_SIZE_MM * px_per_mm)

board_w_px = num_x * sq_px
board_h_px = num_y * sq_px

if board_w_px > page_w_px or board_h_px > page_h_px:
    raise ValueError(
        f"Board ({board_w_px}×{board_h_px}px) does not fit on A4 landscape "
        f"({page_w_px}×{page_h_px}px). Reduce SQUARE_SIZE_MM or INNER_CORNERS."
    )

# Center the board on the page
off_x = (page_w_px - board_w_px) // 2
off_y = (page_h_px - board_h_px) // 2

img = np.ones((page_h_px, page_w_px), dtype=np.uint8) * 255

for row in range(num_y):
    for col in range(num_x):
        if (row + col) % 2 == 0:
            x1 = off_x + col * sq_px
            y1 = off_y + row * sq_px
            img[y1:y1 + sq_px, x1:x1 + sq_px] = 0

cv2.imwrite(OUTPUT, img)

print(f"Generated: {OUTPUT}")
print(f"  Board:   {num_x}×{num_y} squares  ({INNER_CORNERS[0]}×{INNER_CORNERS[1]} inner corners)")
print(f"  Squares: {SQUARE_SIZE_MM}mm  ({sq_px}px each at {DPI} DPI)")
print(f"  Image:   {page_w_px}×{page_h_px}px  (A4 landscape at {DPI} DPI)")
print()
print("PRINTING CHECKLIST:")
print("  1. Print in LANDSCAPE orientation")
print("  2. Set scale to 100% / actual size  (NOT fit-to-page)")
print("  3. Use matte paper — glossy causes reflections that break detection")
print("  4. Mount flat on cardboard or a clipboard — any warp hurts accuracy")
print("  5. After printing, measure one square with a ruler and update")
print("     SQUARE_SIZE_MM in src/calibration/calibrate.py with the real value")
