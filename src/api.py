import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import calculate_diameter
import classify_cable
import edge_proposal
import transform

CALIBRATION_PATH = "data/calibration.npz"
HOMOGRAPHY_PATH = "data/homography.npz"
DISPLAY_SIZE = (3840, 2160)
CROP_X = (0.4, 0.6)
CROP_Y = (0.5, 0.77)
HEIGHT_MM = 48
# How far (in mm) a catalog entry's diameter may be from the measurement to still be
# shown as a candidate match. Several real ACSR conductors share near-identical outer
# diameters despite being different sizes, so this intentionally returns a set of
# plausible matches rather than forcing a single (possibly wrong) best guess.
CLASSIFICATION_TOLERANCE_MM = 0.5

app = FastAPI()

_K, _dist, _H, _bev_size, _pixels_per_mm = transform.load(CALIBRATION_PATH, HOMOGRAPHY_PATH)


class HandlesBody(BaseModel):
    handles: list  # [[top0, bot0], [top1, bot1]], each point [x, y]


def _decode_upload(raw: bytes):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")
    return image


def _encode_png(image) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "Could not encode image")
    return base64.b64encode(buf).decode("ascii")


@app.post("/propose")
async def propose(file: UploadFile = File(...)):
    image = _decode_upload(await file.read())
    roi, undistorted, crop_offset = transform.apply_with_intermediates(
        image, _K, _dist, _H, _bev_size, DISPLAY_SIZE, CROP_X, CROP_Y
    )
    handles = edge_proposal.propose_headless(roi, _pixels_per_mm)
    edges_detected = handles is not None
    if not edges_detected:
        handles = edge_proposal.default_handles(roi)
    return {
        "image_png_base64": _encode_png(roi),
        "handles": handles,
        "edges_detected": edges_detected,
        "undistorted_image_png_base64": _encode_png(undistorted),
        # Let the frontend map points between the two views (including the initial state), without a round-trip.
        "homography": _H.tolist(),
        "homography_inv": np.linalg.inv(_H).tolist(),
        "crop_offset": list(crop_offset),
    }


@app.post("/calculate")
async def calculate(body: HandlesBody):
    diameter_mm = calculate_diameter.calculate(body.handles, _pixels_per_mm, HEIGHT_MM)
    if diameter_mm is None:
        raise HTTPException(422, "Could not calculate a diameter from the given handles")
    return {
        "diameter_mm": round(diameter_mm, 2),
        "candidates": classify_cable.classify(diameter_mm, CLASSIFICATION_TOLERANCE_MM),
    }
