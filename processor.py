import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import easyocr  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from PIL import ImageOps  # type: ignore
from torchvision import transforms  # type: ignore
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore

from train import MnistCNN  # type: ignore

warnings.filterwarnings("ignore", message=".*pin_memory.*")

CANONICAL_WIDTH = 900
CANONICAL_HEIGHT = 1400

EXPECTED_QUESTIONS: List[Tuple[str, int]] = [
    ("Q1.a", 2),
    ("Q1.b", 2),
    ("Q1.c", 2),
    ("Q1.d", 2),
    ("Q1.e", 2),
    ("Q1.f", 2),
    ("Q2.a", 5),
    ("Q2.b", 5),
    ("Q3.a", 5),
    ("Q3.b", 5),
]

QUESTION_CELL_RATIOS: Dict[str, Tuple[float, float, float, float]] = {
    "Q1.a": (0.700, 0.972, 0.085, 0.148),
    "Q1.b": (0.700, 0.972, 0.148, 0.208),
    "Q1.c": (0.700, 0.972, 0.208, 0.267),
    "Q1.d": (0.700, 0.972, 0.267, 0.327),
    "Q1.e": (0.700, 0.972, 0.327, 0.387),
    "Q1.f": (0.700, 0.972, 0.387, 0.446),
    "Q2.a": (0.700, 0.972, 0.446, 0.506),
    "Q2.b": (0.700, 0.972, 0.506, 0.565),
    "Q3.a": (0.700, 0.972, 0.565, 0.625),
    "Q3.b": (0.700, 0.972, 0.625, 0.684),
}

print("Initializing EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

print("Initializing MNIST Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_model = MnistCNN().to(device)
loaded_model = False
for model_path in ["mnist_model.pth"]:
    if not os.path.exists(model_path):
        continue
    try:
        mnist_model.load_state_dict(torch.load(model_path, map_location=device))
        loaded_model = True
        print(f"Loaded MNIST CNN from {model_path}")
        break
    except Exception as model_error:
        print(f"Skipping incompatible model weights from {model_path}: {model_error}")
if not loaded_model:
    print("MNIST CNN weights not found; using random initialization (accuracy will be low).")
mnist_model.eval()

print("Initializing TrOCR Model (this may take a minute)...")
try:
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
    trocr_model.eval()
    print("TrOCR Model loaded successfully.")
except Exception as e:
    print(f"Failed to load TrOCR Model: {e}")
    trocr_model = None
    trocr_processor = None

mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

digit_templates: List[Tuple[str, np.ndarray]] = []


def enhance_image(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(enhanced, None, 6, 6, 7, 21)


def deskew_image(img: np.ndarray, is_webcam: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line = 60 if is_webcam else 100
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=min_line, maxLineGap=15)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < (18 if is_webcam else 10):
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def four_point_transform(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    dst = np.array(
        [
            [0, 0],
            [CANONICAL_WIDTH - 1, 0],
            [CANONICAL_WIDTH - 1, CANONICAL_HEIGHT - 1],
            [0, CANONICAL_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (CANONICAL_WIDTH, CANONICAL_HEIGHT))


def locate_marksheet(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img, (CANONICAL_WIDTH, CANONICAL_HEIGHT))

    img_area = img.shape[0] * img.shape[1]
    best_quad: Optional[np.ndarray] = None
    best_area = 0.0

    target_aspect = float(CANONICAL_WIDTH) / float(CANONICAL_HEIGHT)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Reject small contours so we don't accidentally warp a zoomed-in table fragment.
        if area < img_area * 0.28:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if h <= 0:
                continue
            aspect = float(w) / float(h)
            if aspect < target_aspect * 0.65 or aspect > target_aspect * 1.45:
                continue
            if area > best_area:
                best_quad = approx.reshape(4, 2).astype(np.float32)
                best_area = area

    if best_quad is not None:
        warped = four_point_transform(img, best_quad)
        # Safety: if the warp is near-empty/flat, keep stable full-image resize instead.
        if np.std(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)) > 8.0:
            return warped

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = img[y : y + h, x : x + w]
    return cv2.resize(cropped, (CANONICAL_WIDTH, CANONICAL_HEIGHT))


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    """
    Load an image while respecting EXIF orientation (phone photos).
    OpenCV's cv2.imread ignores EXIF, which makes "straight" images appear rotated/tilted.
    """
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil)
        rgb = np.array(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(path)
        return img


def refine_alignment(img: np.ndarray) -> np.ndarray:
    """
    After perspective normalization, do a small additional deskew.
    This prevents the debug overlay from looking slightly tilted.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 140, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=120, maxLineGap=12)
    if lines is None:
        return img

    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) < 8:
            angles.append(angle)
    if not angles:
        return img

    median_angle = float(np.median(angles))
    # Avoid rotating based on weak/noisy evidence.
    if len(angles) < 10:
        return img
    # Clamp: we only want to correct small residual skew, not rotate aggressively.
    if abs(median_angle) < 0.8:
        return img
    median_angle = float(max(-2.0, min(2.0, median_angle)))

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _bbox_centroid(bbox) -> Tuple[float, float]:
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    return float(sum(xs) / 4.0), float(sum(ys) / 4.0)


def _bbox_y_top_bottom(bbox) -> Tuple[float, float]:
    ys = [float(p[1]) for p in bbox]
    return float(min(ys)), float(max(ys))


def _ocr_find_obt_and_total_anchors(img: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find approximate anchors for:
    - obt_x_center: x center of the printed "Obt." header
    - obt_y_bottom: bottom y of the printed "Obt." header bbox
    - total_y_top: top y of the "Total" / "Total Mark" row
    """
    obt_x_center: Optional[float] = None
    obt_y_bottom: Optional[float] = None
    total_y_top: Optional[float] = None

    try:
        # Use a slightly higher min_size for printed headers
        ocr_results = reader.readtext(img, min_size=15, detail=1)
    except Exception:
        return None, None, None

    for bbox, text, prob in ocr_results:
        t = str(text).strip().lower().replace(".", "").replace(" ", "")
        if not t:
            continue
        cx, cy = _bbox_centroid(bbox)
        y_top, y_bottom = _bbox_y_top_bottom(bbox)

        # "Obt." can be misread in many ways.
        if ("obt" in t or "0bt" in t or "oht" in t or "ob1" in t) and obt_x_center is None:
            # Printed headers are usually in the top 15% of the sheet.
            if cy < img.shape[0] * 0.15:
                obt_x_center = cx
                obt_y_bottom = y_bottom

        # "Total Mark" often comes as ["Total", "Mark"] or one token depending on OCR.
        if "total" in t:
            # Total row is usually in the bottom 40% of the sheet.
            if cy > img.shape[0] * 0.60:
                if total_y_top is None or y_top < total_y_top:
                    total_y_top = y_top

    return obt_x_center, obt_y_bottom, total_y_top


def _extract_projection_centers(projection: np.ndarray, threshold: float, min_gap: int) -> List[int]:
    positions = np.where(projection >= threshold)[0]
    if positions.size == 0:
        return []

    centers: List[int] = []
    start = int(positions[0])
    prev = int(positions[0])
    for pos in positions[1:]:
        pos = int(pos)
        if pos - prev > min_gap:
            centers.append((start + prev) // 2)
            start = pos
        prev = pos
    centers.append((start + prev) // 2)
    return centers


def _merge_close_positions(values: Sequence[int], gap: int) -> List[int]:
    if not values:
        return []
    values_sorted = sorted(int(v) for v in values)
    merged = [values_sorted[0]]
    for v in values_sorted[1:]:
        if abs(v - merged[-1]) <= gap:
            merged[-1] = int((merged[-1] + v) / 2)
        else:
            merged.append(v)
    return merged


def _get_line_positions(binary: np.ndarray, axis: str) -> List[int]:
    h, w = binary.shape
    if axis == "horizontal":
        # Table usually spans from ~10% to ~98% of width.
        # Grid lines are long. Use a kernel that matches typical table column widths.
        kernel_w = max(w // 4, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        line_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        projection = line_img.sum(axis=1)
        if projection.size == 0:
            return []
        threshold = projection.max() * 0.40
        return _extract_projection_centers(projection, threshold, 15)

    # Vertical lines: Q columns, Max column, Obt column.
    kernel_h = max(h // 6, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    line_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    projection = line_img.sum(axis=0)
    if projection.size == 0:
        return []
    threshold = projection.max() * 0.40
    return _extract_projection_centers(projection, threshold, 15)


def detect_table_cells(img: np.ndarray, is_webcam: bool = False) -> Tuple[List[int], Optional[int], Optional[int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_webcam:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    horizontal_lines = _get_line_positions(binary, "horizontal")
    vertical_lines = _get_line_positions(binary, "vertical")

    max_col_x = vertical_lines[-3] if len(vertical_lines) >= 4 else None
    obt_col_x = vertical_lines[-2] if len(vertical_lines) >= 3 else None
    return horizontal_lines, max_col_x, obt_col_x


def _intervals_from_lines(lines: Sequence[int], image_limit: int, min_size: int) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    if len(lines) < 2:
        return intervals

    for left, right in zip(lines[:-1], lines[1:]):
        start = int(left)
        end = int(right)
        if end - start >= min_size:
            intervals.append((start, end))

    if not intervals and image_limit > 0:
        intervals.append((0, image_limit))
    return intervals


def _question_rect(question_name: str, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    left_ratio, right_ratio, top_ratio, bottom_ratio = QUESTION_CELL_RATIOS[question_name]
    left = int(left_ratio * image_width)
    right = int(right_ratio * image_width)
    top = int(top_ratio * image_height)
    bottom = int(bottom_ratio * image_height)
    return left, right, top, bottom


def _snap_to_nearest_line(value: int, lines: Sequence[int], tolerance: int = 45) -> int:
    if not lines:
        return value
    nearest = min(lines, key=lambda line: abs(int(line) - value))
    if abs(int(nearest) - value) <= tolerance:
        return int(nearest)
    return value


def _extract_table_lines(img: np.ndarray, is_webcam: bool = False) -> Tuple[List[int], List[int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_webcam:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return _get_line_positions(binary, "horizontal"), _get_line_positions(binary, "vertical")


def _pick_obt_column_bounds(
    v_lines: Sequence[int], obt_x_center: Optional[float], img_width: int
) -> Tuple[int, int]:
    if not v_lines:
        return int(img_width * 0.72), int(img_width * 0.98)

    v_sorted = sorted(set(int(x) for x in v_lines))

    # If we can locate the printed "Obt." header, choose the nearest vertical lines around it.
    if obt_x_center is not None:
        left_candidates = [x for x in v_sorted if x <= obt_x_center]
        right_candidates = [x for x in v_sorted if x >= obt_x_center]
        if left_candidates and right_candidates:
            left = max(left_candidates)
            right = min(right_candidates)
            if right - left >= 60:
                return left, right

    # Fallback: the Obt. column is typically the right-most column, bounded by the last two vertical lines.
    if len(v_sorted) >= 2:
        return v_sorted[-2], v_sorted[-1]
    return int(img_width * 0.72), int(img_width * 0.98)


def _table_vertical_lines(binary: np.ndarray, y1: int, y2: int) -> List[int]:
    """
    Detect vertical grid lines inside the marks table.

    Global projection-based detection often only returns the outer border lines, because inner
    grid lines do not span the full image height. This ROI-based contour approach is more stable.
    """
    h, w = binary.shape[:2]
    y1 = max(0, int(y1))
    y2 = min(h, int(y2))
    if y2 - y1 < 50:
        return []

    roi = binary[y1:y2, :]
    roi_h = roi.shape[0]

    xs: List[int] = []

    # 1) Morphology-based extraction (fast when lines are continuous).
    kernel_h = max(int(roi_h * 0.45), 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    line_img = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < int(roi_h * 0.28):
            continue
        if cw > max(18, int(w * 0.03)):
            continue
        xs.append(int(x + cw / 2))

    xs = _merge_close_positions(xs, gap=12)

    # 2) Hough fallback (handles broken/dashed lines due to thresholding/shadows).
    if len(xs) < 4:
        edges = cv2.Canny(roi, 40, 140, apertureSize=3)
        min_len = max(int(roi_h * 0.25), 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 110, minLineLength=min_len, maxLineGap=18)
        if lines is not None:
            for line in lines:
                x1p, y1p, x2p, y2p = (int(v) for v in line[0])
                if abs(x2p - x1p) <= 8 and abs(y2p - y1p) >= int(roi_h * 0.22):
                    xs.append(int((x1p + x2p) / 2))
        xs = _merge_close_positions(xs, gap=12)

    return xs


def _roi_horizontal_lines(binary: np.ndarray, x1: int, x2: int) -> List[int]:
    roi = binary[:, max(0, x1) : min(binary.shape[1], x2)]
    if roi.size == 0:
        return []

    h, w = roi.shape
    # Relaxed kernels for broken/faint lines
    kernel_w = max(int(w * 0.35), 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    line_img = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * 0.30:
            continue
        ys.append(int(y + ch / 2))

    ys = _merge_close_positions(ys, gap=10)

    # Hough fallback for faint/broken separators - more aggressive.
    if len(ys) < 11:
        edges = cv2.Canny(roi, 30, 100, apertureSize=3)
        min_len = max(int(w * 0.30), 50)
        # Lower threshold for better sensitivity to faint lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=min_len, maxLineGap=25)
        if lines is not None:
            for line in lines:
                x1p, y1p, x2p, y2p = (int(v) for v in line[0])
                if abs(y2p - y1p) <= 10 and abs(x2p - x1p) >= min_len:
                    ys.append(int((y1p + y2p) / 2))
        ys = _merge_close_positions(ys, gap=10)
    return ys


def _best_11_line_sequence(lines: Sequence[int]) -> Optional[List[int]]:
    if len(lines) < 11:
        return None

    lines_sorted = sorted(lines)
    best_seq = None
    best_score = None

    for start in range(0, len(lines_sorted) - 10):
        seq = lines_sorted[start : start + 11]
        diffs = [seq[i + 1] - seq[i] for i in range(10)]
        if any(d <= 8 for d in diffs):
            continue
        mean = float(sum(diffs)) / 10.0
        var = float(sum((d - mean) ** 2 for d in diffs)) / 10.0
        # Prefer consistent spacing; also prefer typical cell height range.
        penalty = 0.0
        if mean < 28 or mean > 85:
            penalty = abs(mean - 48.0)
        score = var + penalty * 10.0
        if best_score is None or score < best_score:
            best_score = score
            best_seq = seq

    return best_seq


def _best_11_line_sequence_anchored(
    lines: Sequence[int],
    img_h: int,
    prefer_start_y: Optional[int],
    max_end_y: Optional[int],
) -> Optional[List[int]]:
    """
    Choose the best 11 horizontal separators for the 10 Obt cells.

    Critical requirement: must not "shift" when a top cell is blank.
    So we bias toward the earliest plausible grid sequence (just under the Obt header),
    rather than anything influenced by handwriting.
    """
    if len(lines) < 11:
        return None

    lines_sorted = sorted(int(y) for y in lines)

    # Expected region for the first separator below the header row in the normalized sheet.
    # This keeps us from accidentally starting the sequence at Q1.b when OCR anchors are wrong.
    expected_start = prefer_start_y if prefer_start_y is not None else int(img_h * 0.055)
    # Don't hard-reject based on start position; different photos/perspective can shift the table.
    # We'll use a soft penalty instead.
    start_min = int(img_h * 0.01)
    start_max = int(img_h * 0.35)

    best_seq: Optional[List[int]] = None
    best_score: Optional[float] = None

    for start in range(0, len(lines_sorted) - 10):
        seq = lines_sorted[start : start + 11]
        if max_end_y is not None and seq[-1] > int(max_end_y):
            continue

        # Very top/bottom sequences are unlikely (noise), but keep this permissive.
        if seq[0] < start_min or seq[0] > start_max:
            start_outside = True
        else:
            start_outside = False

        diffs = [seq[i + 1] - seq[i] for i in range(10)]
        if any(d <= 8 for d in diffs):
            continue

        mean = float(sum(diffs)) / 10.0
        var = float(sum((d - mean) ** 2 for d in diffs)) / 10.0

        # Penalize weird cell heights.
        height_penalty = 0.0
        # Expected cell height in normalized 900x1400 sheet is around 60-75px.
        if mean < 45 or mean > 90:
            height_penalty = abs(mean - 68.0)

        # Bias towards the expected header-adjacent start (soft).
        start_penalty = abs(float(seq[0] - expected_start)) / float(max(img_h, 1))
        if start_outside:
            start_penalty += 0.25 # Increased penalty for outside expected range

        # Total score: variance (consistency) + height penalty + start position penalty.
        score = var + height_penalty * 20.0 + start_penalty * 15000.0
        if best_score is None or score < best_score:
            best_score = score
            best_seq = seq

    # As a fallback, return the purely spacing-based selection.
    return best_seq or _best_11_line_sequence(lines_sorted)


def _derive_obt_row_boundaries_from_roi(
    binary: np.ndarray,
    obt_left: int,
    obt_right: int,
    obt_y_bottom: Optional[float],
    total_y_top: Optional[float],
) -> Optional[List[int]]:
    img_h = binary.shape[0]
    img_w = binary.shape[1]
    
    # Expand ROI to include "Max." column for more stable horizontal line detection.
    # The grid lines span across Max and Obt columns.
    search_left = max(0, int(img_w * 0.45))
    search_right = min(img_w, int(img_w * 0.98))
    
    # Detect horizontal separators within the wider table ROI.
    roi_lines_all = sorted(set(_roi_horizontal_lines(binary, search_left, search_right)))
    
    # Fallback to global lines if ROI detection is too sparse.
    if len(roi_lines_all) < 11:
        roi_lines_all = sorted(set(_merge_close_positions(_get_line_positions(binary, "horizontal"), gap=10)))

    # Use OCR only as a soft hint. If OCR is wrong/missing, we still pick the top-most plausible grid sequence.
    prefer_start_y = None
    if obt_y_bottom is not None:
        # First separator is typically shortly below the "Obt." header text box.
        prefer_start_y = int(float(obt_y_bottom) + 12.0)

    max_end_y = None
    if total_y_top is not None:
        max_end_y = int(float(total_y_top) - 6.0)

    seq = _best_11_line_sequence_anchored(roi_lines_all, img_h, prefer_start_y=prefer_start_y, max_end_y=max_end_y)
    if seq is None:
        return None

    # Final sanity: ensure it's increasing and within bounds.
    if len(seq) != 11 or any(seq[i + 1] <= seq[i] for i in range(10)):
        return None
    return seq


def _expected_obt_row_boundaries(img_h: int) -> List[int]:
    first_top_ratio = QUESTION_CELL_RATIOS["Q1.a"][2]
    boundaries = [int(first_top_ratio * img_h)]
    for question_name, _ in EXPECTED_QUESTIONS:
        boundaries.append(int(QUESTION_CELL_RATIOS[question_name][3] * img_h))
    return boundaries


def _snap_boundaries_to_detected(
    expected: Sequence[int],
    detected: Optional[Sequence[int]],
    tol: int,
) -> List[int]:
    if not detected:
        return [int(y) for y in expected]

    detected_sorted = sorted(int(y) for y in detected)
    snapped: List[int] = []
    for y_exp in expected:
        nearest = min(detected_sorted, key=lambda y: abs(y - int(y_exp)))
        if abs(nearest - int(y_exp)) <= tol:
            snapped.append(int(nearest))
        else:
            snapped.append(int(y_exp))

    fixed = [snapped[0]]
    # In normalized sheets, row height is roughly ~70-85px. Keep a strong lower bound
    # so noisy line snaps cannot collapse a row and break OCR.
    min_gap = max(42, int(tol * 0.9))
    for y in snapped[1:]:
        fixed.append(max(int(y), fixed[-1] + min_gap))
    return fixed


def _aligned_expected_row_boundaries(
    img_h: int,
    detected: Optional[Sequence[int]],
) -> List[int]:
    """
    Use template row geometry (fixed marksheet) and only apply a small global vertical offset
    inferred from detected lines. This prevents per-row drift and row collapsing.
    """
    expected = _expected_obt_row_boundaries(img_h)
    if not detected:
        return expected

    detected_sorted = sorted(int(y) for y in detected)
    offsets: List[int] = []
    for y_exp in expected:
        nearest = min(detected_sorted, key=lambda y: abs(y - y_exp))
        if abs(nearest - y_exp) <= max(48, int(img_h * 0.045)):
            offsets.append(int(nearest - y_exp))

    if len(offsets) < 5:
        return expected

    delta = int(np.median(np.array(offsets, dtype=np.int32)))
    delta = max(-60, min(60, delta))
    shifted = [max(0, min(img_h - 1, y + delta)) for y in expected]
    return shifted


def _snap_expected_lines_locally(
    expected: Sequence[int],
    detected: Optional[Sequence[int]],
    search_radius: int,
    min_gap: int,
) -> List[int]:
    """
    Snap each expected boundary to the nearest detected horizontal line independently.
    This is more stable than applying one global offset when a single row is misdetected.
    """
    if not detected:
        return [int(y) for y in expected]

    detected_sorted = sorted(int(y) for y in detected)
    snapped: List[int] = []
    prev_y: Optional[int] = None

    for idx, y_exp in enumerate(expected):
        candidates = [y for y in detected_sorted if abs(y - int(y_exp)) <= search_radius]
        if candidates:
            y = min(candidates, key=lambda v: abs(v - int(y_exp)))
        else:
            y = int(y_exp)

        if prev_y is not None:
            y = max(y, prev_y + min_gap)
        snapped.append(int(y))
        prev_y = int(y)

    return snapped


def _snap_obt_column_bounds_locally(
    expected_left: int,
    expected_right: int,
    detected: Sequence[int],
    tol: int,
) -> Tuple[int, int]:
    if not detected:
        return expected_left, expected_right

    detected_sorted = sorted(int(x) for x in detected)
    left_candidates = [x for x in detected_sorted if abs(x - expected_left) <= tol]
    right_candidates = [x for x in detected_sorted if abs(x - expected_right) <= tol]

    left = min(left_candidates, key=lambda x: abs(x - expected_left)) if left_candidates else expected_left
    right = min(right_candidates, key=lambda x: abs(x - expected_right)) if right_candidates else expected_right

    if right - left < 140:
        return expected_left, expected_right
    return int(left), int(right)


def detect_obt_cells(img: np.ndarray, is_webcam: bool = False) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Detect the 10 Obt. cells (below the printed 'Obt.' header and above the Total row).
    Returns list of 10 rectangles: (x1, y1, x2, y2) in image coordinates.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block = 31
    c = 9
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c
    )

    obt_x_center, obt_y_bottom, total_y_top = _ocr_find_obt_and_total_anchors(img)

    img_h, img_w = img.shape[:2]
    default_left = int(img_w * 0.700)
    default_right = int(img_w * 0.972)
    obt_left, obt_right = default_left, default_right

    y2_table = int((total_y_top or (img_h * 0.74)) + 40)
    v_lines = _table_vertical_lines(binary, y1=int(img_h * 0.02), y2=y2_table)
    if len(v_lines) < 4:
        v_lines = _get_line_positions(binary, "vertical")

    picked_left, picked_right = _pick_obt_column_bounds(v_lines, obt_x_center, img_w)
    picked_w = picked_right - picked_left
    picked_center = (picked_left + picked_right) / 2.0
    if (
        picked_w >= int(img_w * 0.15)
        and picked_w <= int(img_w * 0.45)
        and picked_center >= float(img_w) * 0.60
    ):
        obt_left, obt_right = picked_left, picked_right

    obt_left, obt_right = _snap_obt_column_bounds_locally(
        obt_left,
        obt_right,
        v_lines,
        tol=max(20, int(img_w * 0.03)),
    )

    # Robust horizontal line detection
    search_left = max(0, int(img_w * 0.45))
    search_right = min(img_w, int(img_w * 0.98))
    roi_lines = sorted(set(_roi_horizontal_lines(binary, search_left, search_right)))
    
    # Define the table's vertical span for the 10 Obt cells.
    # Start: just below "Obt." header. End: just above "Total" row.
    y_start_expected = int(img_h * 0.085)
    y_end_expected = int(img_h * 0.684)
    
    if obt_y_bottom is not None:
        y_start_expected = int(obt_y_bottom + 5)
    if total_y_top is not None:
        y_end_expected = int(total_y_top - 5)

    # Snap start/end to nearest detected lines if they are close.
    y_start = _snap_to_nearest_line(y_start_expected, roi_lines, tolerance=40)
    y_end = _snap_to_nearest_line(y_end_expected, roi_lines, tolerance=40)
    
    # If we found a good sequence of lines, use it.
    seq = _best_11_line_sequence_anchored(roi_lines, img_h, prefer_start_y=y_start, max_end_y=y_end + 20)
    
    if seq is not None and len(seq) == 11:
        row_lines = seq
    else:
        # Interpolate 11 lines between y_start and y_end.
        row_lines = [int(y_start + i * (y_end - y_start) / 10) for i in range(11)]
        # Snap each interpolated line to the nearest detected line.
        row_lines = _snap_expected_lines_locally(row_lines, roi_lines, search_radius=25, min_gap=40)

    if len(row_lines) != 11:
        return None

    pad_x = max(1, int((obt_right - obt_left) * 0.01))
    rects: List[Tuple[int, int, int, int]] = []
    for i in range(10):
        y_top = int(row_lines[i])
        y_bottom = int(row_lines[i + 1])
        cell_h = max(1, y_bottom - y_top)
        
        # Minimal vertical padding to avoid missing digits that touch lines
        pad_y = max(1, int(cell_h * 0.02))

        x1 = max(0, obt_left + pad_x)
        x2 = min(img_w, obt_right - pad_x)
        y1 = max(0, y_top + pad_y)
        y2 = min(img_h, y_bottom - pad_y)
        if x2 <= x1 or y2 <= y1:
            # Fallback for collapsed cells
            rects.append((x1, y_top + 2, x2, y_bottom - 2))
        else:
            rects.append((x1, y1, x2, y2))

    return rects if len(rects) == 10 else None


def _calibrated_question_rect(
    question_name: str,
    image_width: int,
    image_height: int,
    h_lines: Sequence[int],
    v_lines: Sequence[int],
) -> Tuple[int, int, int, int]:
    left, right, top, bottom = _question_rect(question_name, image_width, image_height)
    calibrated_left = _snap_to_nearest_line(left, v_lines, tolerance=35)
    calibrated_right = _snap_to_nearest_line(right, v_lines, tolerance=35)
    calibrated_top = _snap_to_nearest_line(top, h_lines, tolerance=28)
    calibrated_bottom = _snap_to_nearest_line(bottom, h_lines, tolerance=28)

    if calibrated_right - calibrated_left < 80:
        calibrated_left, calibrated_right = left, right
    if calibrated_bottom - calibrated_top < 30:
        calibrated_top, calibrated_bottom = top, bottom

    return calibrated_left, calibrated_right, calibrated_top, calibrated_bottom


def extract_obt_cells(
    img: np.ndarray, h_lines: List[int], obt_col_x: Optional[int]
) -> List[Tuple[np.ndarray, int, int, int, int]]:
    cells: List[Tuple[np.ndarray, int, int, int, int]] = []
    for question_name, _ in EXPECTED_QUESTIONS:
        left, right, top, bottom = _question_rect(question_name, img.shape[1], img.shape[0])
        crop, rect = _crop_with_padding(img, top, bottom, left, right, 0.03, 0.12)
        x, y, w, h = rect
        cells.append((crop, x, y, w, h))
    return cells


def is_likely_dash(crop_img: np.ndarray) -> bool:
    if crop_img.size == 0:
        return True
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if ink_ratio < 0.015:
        return True

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [contour for contour in contours if cv2.contourArea(contour) > 18]
    if not valid:
        return True

    x, y, w, h = cv2.boundingRect(max(valid, key=cv2.contourArea))
    return h < 8 or (w > 2.2 * h and h < crop_img.shape[0] * 0.35)


def is_empty_cell(crop_img: np.ndarray) -> bool:
    """
    Detect truly empty cells (no handwriting) without confusing thin digits like '1' as "empty".
    We focus on the right half because marks are usually written like "02", "01", "5", etc.
    """
    if crop_img.size == 0:
        return True

    # Delegate to a stricter presence detector that ignores most background noise.
    return not _has_confident_mark_presence(crop_img)


def _prepare_digit_mask(crop_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Illumination normalization (phone shadows): estimate background and emphasize dark ink.
    h0, w0 = gray.shape[:2]
    k = max(15, (min(h0, w0) // 6) | 1)  # odd kernel, proportional to crop size
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel, iterations=1)
    diff = cv2.subtract(background, gray)  # handwriting becomes bright
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    h, w = thresh.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 2, 20), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 20)))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Only subtract lines if they are not too thick (to avoid removing digits)
    clean = thresh.copy()
    if np.count_nonzero(horizontal_lines) < thresh.size * 0.15:
        clean = cv2.subtract(clean, horizontal_lines)
    if np.count_nonzero(vertical_lines) < thresh.size * 0.15:
        clean = cv2.subtract(clean, vertical_lines)

    # Avoid aggressive dilation: it turns background speckles into "fake digits" and causes false positives.
    clean = cv2.medianBlur(clean, 3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Remove residual cell borders at the crop edges so grid lines do not become "digits".
    # Keep this conservative; marks often touch the borders (especially '1').
    edge_y = max(1, int(h * 0.10))
    edge_x = max(1, int(w * 0.05))
    clean[:edge_y, :] = 0
    clean[h - edge_y :, :] = 0
    clean[:, :edge_x] = 0
    clean[:, w - edge_x :] = 0
    return clean


def clean_crop_for_ocr(crop_img: np.ndarray) -> np.ndarray:
    clean = _prepare_digit_mask(crop_img)
    clean = cv2.bitwise_not(clean)
    clean_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(clean_bgr, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def preprocess_for_mnist(crop_img: np.ndarray) -> np.ndarray:
    # If we're already given a binary-ish component (e.g., from segmentation),
    # don't re-threshold it; _prepare_digit_mask is tuned for full cell crops and
    # can wipe out thin strokes like '1' when applied twice.
    if len(crop_img.shape) == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()

    unique = np.unique(gray)
    if unique.size <= 6 and int(gray.max()) >= 200 and int(gray.min()) <= 10:
        mask = (gray > 0).astype(np.uint8) * 255
        # Slightly thicken very thin strokes so they survive resizing.
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    else:
        mask = _prepare_digit_mask(crop_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Lower threshold so thin '1' strokes aren't discarded.
    valid = [contour for contour in contours if cv2.contourArea(contour) > 8]
    if not valid:
        return np.zeros((28, 28), dtype=np.uint8)

    x1 = min(cv2.boundingRect(contour)[0] for contour in valid)
    y1 = min(cv2.boundingRect(contour)[1] for contour in valid)
    x2 = max(cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in valid)
    y2 = max(cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in valid)
    digit_crop = mask[y1:y2, x1:x2]
    if digit_crop.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    h, w = digit_crop.shape
    max_dim = max(h, w)
    padding = max(4, max_dim // 4)
    canvas = np.zeros((max_dim + 2 * padding, max_dim + 2 * padding), dtype=np.uint8)
    y_offset = (canvas.shape[0] - h) // 2
    x_offset = (canvas.shape[1] - w) // 2
    canvas[y_offset : y_offset + h, x_offset : x_offset + w] = digit_crop
    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    return cv2.dilate(resized, np.ones((2, 2), np.uint8), iterations=1)


def _build_template_vector(processed: np.ndarray) -> Optional[np.ndarray]:
    if processed.size == 0 or np.sum(processed) == 0:
        return None
    vector = processed.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def load_digit_templates(template_dir: str = "labeled_digits") -> None:
    global digit_templates
    digit_templates = []

    if not os.path.isdir(template_dir):
        return

    for label in sorted(os.listdir(template_dir)):
        label_dir = os.path.join(template_dir, label)
        if not os.path.isdir(label_dir) or not label.isdigit():
            continue

        for filename in os.listdir(label_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            image = cv2.imread(os.path.join(label_dir, filename))
            if image is None:
                continue
            processed = preprocess_for_mnist(image)
            vector = _build_template_vector(processed)
            if vector is not None:
                digit_templates.append((label, vector))


def predict_mark_template(crop_img: np.ndarray) -> Tuple[Optional[str], float]:
    if not digit_templates:
        return None, 0.0

    processed = preprocess_for_mnist(crop_img)
    vector = _build_template_vector(processed)
    if vector is None:
        return None, 0.0

    best_label = None
    best_score = -1.0
    for label, template_vector in digit_templates:
        score = float(np.dot(vector, template_vector))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


def predict_mark_template_restricted(crop_img: np.ndarray, allowed: Sequence[str]) -> Tuple[Optional[str], float]:
    allowed_set = set(str(a) for a in allowed)
    if not digit_templates or not allowed_set:
        return None, 0.0

    processed = preprocess_for_mnist(crop_img)
    vector = _build_template_vector(processed)
    if vector is None:
        return None, 0.0

    best_label = None
    best_score = -1.0
    for label, template_vector in digit_templates:
        if label not in allowed_set:
            continue
        score = float(np.dot(vector, template_vector))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


load_digit_templates()


def predict_mark_cnn(crop_img: np.ndarray) -> Tuple[str, float]:
    processed = preprocess_for_mnist(crop_img)
    if np.sum(processed) == 0:
        return "-", 0.0

    tensor = mnist_transform(Image.fromarray(processed)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = mnist_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_pred = torch.max(probabilities, dim=1)

    return str(int(top_pred.item())), float(top_prob.item())


def predict_mark_cnn_restricted(crop_img: np.ndarray, allowed: Sequence[str]) -> Tuple[str, float]:
    allowed_digits = [int(a) for a in allowed if str(a).isdigit()]
    if not allowed_digits:
        return "-", 0.0

    processed = preprocess_for_mnist(crop_img)
    if np.sum(processed) == 0:
        return "-", 0.0

    tensor = mnist_transform(Image.fromarray(processed)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = mnist_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)

    best_d = allowed_digits[0]
    best_p = float(probabilities[best_d].item())
    for d in allowed_digits[1:]:
        p = float(probabilities[d].item())
        if p > best_p:
            best_d = d
            best_p = p
    return str(int(best_d)), float(best_p)


def _normalize_ocr_text(text: str) -> str:
    replacements = {
        "O": "0",
        "o": "0",
        "l": "1",
        "I": "1",
        "|": "1",
        "/": "1",
        "\\": "1",
        "!": "1",
        "S": "5",
        "s": "5",
        "Z": "2",
        "z": "2",
    }
    cleaned = "".join(replacements.get(char, char) for char in text.strip())
    if cleaned in {"-", "–", "—"}:
        return "-"
    return "".join(char for char in cleaned if char.isdigit() or char == "-")


def predict_mark_easyocr(crop_img: np.ndarray) -> Tuple[str, float]:
    try:
        clean_crop = clean_crop_for_ocr(crop_img)
        h, w = clean_crop.shape[:2]
        enlarged = cv2.resize(clean_crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(enlarged, allowlist="0123456789-", min_size=5, detail=1)
        if not results:
            return "-", 0.0
        best = max(results, key=lambda item: item[2])
        normalized = _normalize_ocr_text(str(best[1]))
        return normalized or "-", float(best[2])
    except Exception:
        return "-", 0.0


def predict_mark_trocr(crop_img: np.ndarray, max_mark: int) -> Tuple[str, float]:
    """
    Use Transformer-based OCR for handwriting recognition.
    """
    if trocr_model is None or trocr_processor is None:
        return "-", 0.0

    try:
        # Enhancement for OCR: contrast stretching
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        p2, p98 = np.percentile(gray, (2, 98))
        rescaled = np.clip((gray - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(rescaled, cv2.COLOR_GRAY2BGR)
        
        pil_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
        pixel_values = trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values, max_new_tokens=5)

        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if not text or text.lower() == "none":
            return "-", 0.0
        normalized = _normalize_ocr_text(text)
        value = _extract_digits_from_text(normalized, max_mark)

        if value and value.isdigit():
            return value, 0.95
        return "-", 0.0
    except Exception as e:
        print(f"TrOCR prediction failed: {e}")
        return "-", 0.0


def _extract_digits_from_text(text: str, max_mark: int) -> Optional[str]:
    normalized = _normalize_ocr_text(text)
    if normalized == "-":
        return "-"

    digits = "".join(char for char in normalized if char.isdigit())
    if not digits:
        return None

    if len(digits) >= 2:
        value = int(digits[:2])
        if value <= max_mark:
            return str(value)
        if digits[0] == "0" and int(digits[1]) <= max_mark:
            return str(int(digits[1]))

    single = int(digits[-1])
    if single <= max_mark:
        return str(single)
    return None


def _split_wide_component(component: np.ndarray) -> List[np.ndarray]:
    if component.size == 0:
        return []
    projection = component.sum(axis=0)
    if projection.size < 8:
        return [component]

    middle = projection.shape[0] // 2
    window_start = max(1, middle - projection.shape[0] // 5)
    window_end = min(projection.shape[0] - 1, middle + projection.shape[0] // 5)
    split = int(np.argmin(projection[window_start:window_end]) + window_start)
    if split <= 2 or split >= projection.shape[0] - 2:
        return [component]

    left = component[:, :split]
    right = component[:, split:]
    if left.sum() == 0 or right.sum() == 0:
        return [component]
    return [left, right]


def _segment_digit_images(crop_img: np.ndarray) -> List[np.ndarray]:
    mask = _prepare_digit_mask(crop_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # Allow very thin '1' strokes (w=1-2).
        if area < 18 or h < 10 or w < 1:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda item: item[0])
    digits: List[np.ndarray] = []
    for x, y, w, h in boxes:
        component = mask[max(y - 2, 0) : min(y + h + 2, mask.shape[0]), max(x - 2, 0) : min(x + w + 2, mask.shape[1])]
        if w > h * 1.4:
            digits.extend(_split_wide_component(component))
        else:
            digits.append(component)

    return [digit for digit in digits if digit.size > 0 and np.count_nonzero(digit) > 24]


def _extract_right_digit_crop(crop_img: np.ndarray) -> Optional[np.ndarray]:
    mask = _prepare_digit_mask(crop_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    width = mask.shape[1]
    min_h = max(10, int(mask.shape[0] * 0.30))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # Tolerant for thin '1' and faint strokes (phone photos).
        if h < min_h or w < 1 or area < 10:
            continue
        if x + w / 2 < width * 0.45:
            continue
        candidate_boxes.append((x, y, w, h))

    if not candidate_boxes:
        return None

    # Pick the rightmost digit component and only group very nearby fragments
    # (to avoid accidentally including the leading "0" in "02"/"01").
    rx, ry, rw, rh = max(candidate_boxes, key=lambda b: b[0] + b[2] / 2.0)
    right_center = float(rx + rw / 2.0)
    grouped = []
    for x, y, w, h in candidate_boxes:
        center = float(x + w / 2.0)
        if abs(center - right_center) <= width * 0.10:
            grouped.append((x, y, w, h))

    x1 = min(x for x, y, w, h in grouped)
    y1 = min(y for x, y, w, h in grouped)
    x2 = max(x + w for x, y, w, h in grouped)
    y2 = max(y + h for x, y, w, h in grouped)

    pad = 3
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask.shape[1], x2 + pad)
    y2 = min(mask.shape[0], y2 + pad)
    digit_mask = mask[y1:y2, x1:x2]
    if digit_mask.size == 0 or np.count_nonzero(digit_mask) < 25:
        return None
    if digit_mask.shape[0] < 16 or digit_mask.shape[1] < 8:
        return None
    return cv2.cvtColor(digit_mask, cv2.COLOR_GRAY2BGR)


def _has_leading_zero_component(crop_img: np.ndarray) -> bool:
    mask = _prepare_digit_mask(crop_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    width = mask.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if x + w / 2 > width * 0.48:
            continue
        if area < 80 or w < 12 or h < 14:
            continue
        aspect_ratio = w / max(h, 1)
        if 0.45 <= aspect_ratio <= 1.4:
            return True
    return False


def _has_confident_mark_presence(crop_img: np.ndarray) -> bool:
    mask = _prepare_digit_mask(crop_img)
    h, w = mask.shape
    if mask.size == 0:
        return False

    # More relaxed ink ratio for faint handwriting
    ink_ratio = float(np.count_nonzero(mask)) / float(max(mask.size, 1))
    if ink_ratio < 0.0035:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    min_ch = max(10, int(h * 0.20))
    right_components = 0
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))

        # Reject only very tiny noise
        if area < 10:
            continue

        # Digit-ish vertical extent.
        if ch < min_ch:
            continue

        # More tolerant mark-writing zone.
        if x + cw / 2 >= w * 0.25:
            right_components += 1
            if area >= 20:
                return True

        # Strong blob anywhere in the cell.
        if area >= 45:
            return True

    return right_components >= 1


def _predict_single_digit(component: np.ndarray, allowed: Sequence[str]) -> Optional[str]:
    if component.size == 0:
        return None

    component_bgr = (
        cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        if len(component.shape) == 2
        else component
    )
    gray = cv2.cvtColor(component_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    nz = int(np.count_nonzero(gray))
    # Guard against tiny artifacts being forced into a digit class.
    if h < 10 or w < 2 or nz < 16:
        return None

    # CNN tends to be much better on thin digits like '1' where templates can be misleading.
    cnn_pred, cnn_conf = predict_mark_cnn_restricted(component_bgr, allowed)
    
    # Skip templates if folder is empty or non-existent
    template_pred, template_score = None, 0.0
    if digit_templates:
        template_pred, template_score = predict_mark_template_restricted(component_bgr, allowed)

    if cnn_pred.isdigit() and cnn_conf >= 0.88:
        return cnn_pred
    if template_pred is not None and template_score >= 0.90:
        return template_pred
    if (
        cnn_pred.isdigit()
        and template_pred is not None
        and cnn_pred == template_pred
        and cnn_conf >= 0.65
        and template_score >= 0.65
    ):
        return cnn_pred

    ocr_pred, ocr_conf = predict_mark_easyocr(component_bgr)
    if ocr_pred and ocr_conf >= 0.45:
        extracted = _extract_digits_from_text(ocr_pred, max_mark=max(int(a) for a in allowed if str(a).isdigit()))
        if extracted is not None and extracted.isdigit():
            return extracted[-1]
    return None


def predict_mark_combined(crop_img: np.ndarray, max_mark: int) -> str:
    # If teacher draws a dash for "not attempted", keep treating it as blank.
    if is_likely_dash(crop_img):
        return "-"
    
    # 1) Try TrOCR first as it's the most powerful model.
    # We run it BEFORE the strict presence check because TrOCR is good at 
    # seeing digits in faint handwriting that simple CV might miss.
    trocr_val, trocr_conf = predict_mark_trocr(crop_img, max_mark)
    if trocr_val.isdigit() and 1 <= int(trocr_val) <= max_mark and trocr_conf >= 0.85:
        return trocr_val

    # 2) Strict presence check for other models.
    if not _has_confident_mark_presence(crop_img):
        # Even if TrOCR found something but with low confidence, if the cell is 
        # visually empty, trust the empty check.
        return "-"

    # Output domain: marks are 1..max_mark.
    allowed_out = [str(i) for i in range(1, max_mark + 1)]

    # 3) Primary CNN check.
    digit_images = _segment_digit_images(crop_img)
    if digit_images:
        rightmost = digit_images[-1]
        pred = _predict_single_digit(rightmost, allowed_out)
        if pred is not None and pred.isdigit() and 1 <= int(pred) <= max_mark:
            return pred

    # 4) Fallback to TrOCR even if low confidence if CNN failed.
    if trocr_val.isdigit() and 1 <= int(trocr_val) <= max_mark:
        return trocr_val

    # 5) Fallback: whole-cell OCR with EasyOCR.
    whole_ocr_pred, whole_ocr_conf = predict_mark_easyocr(crop_img)
    whole_ocr_value = _extract_digits_from_text(whole_ocr_pred, max_mark)
    if (
        whole_ocr_value is not None
        and whole_ocr_value.isdigit()
        and 1 <= int(whole_ocr_value) <= max_mark
        and whole_ocr_conf >= 0.35
    ):
        return whole_ocr_value

    return "-"


def _crop_with_padding(
    img: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    cell_h = max(bottom - top, 1)
    cell_w = max(right - left, 1)
    pad_x = max(6, int(cell_w * pad_x_ratio))
    pad_y = max(4, int(cell_h * pad_y_ratio))

    x1 = max(0, left + pad_x)
    x2 = min(img.shape[1], right - pad_x)
    y1 = max(0, top + pad_y)
    y2 = min(img.shape[0], bottom - pad_y)

    if x2 <= x1:
        x1, x2 = max(0, left + 2), min(img.shape[1], right - 2)
    if y2 <= y1:
        y1, y2 = max(0, top + 2), min(img.shape[0], bottom - 2)

    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, max(x2 - x1, 1), max(y2 - y1, 1))


def process_image(img_path: str, session_folder: str = None, is_webcam: bool = False) -> Tuple[List[Dict], int, str]:
    img = load_image_bgr(img_path)
    if img is None:
        raise Exception("Failed to load image")

    img = deskew_image(img, is_webcam=is_webcam)
    normalized = locate_marksheet(img)
    normalized = enhance_image(normalized)
    normalized = refine_alignment(normalized)

    debug_img = normalized.copy()
    extracted_data: List[Dict[str, str]] = []

    obt_rects = detect_obt_cells(normalized, is_webcam=is_webcam)

    for idx, (question_name, q_max) in enumerate(EXPECTED_QUESTIONS):
        if obt_rects is not None:
            x1, y1, x2, y2 = obt_rects[idx]
            crop_img = normalized[y1:y2, x1:x2]
            x, y, w, h = x1, y1, max(1, x2 - x1), max(1, y2 - y1)
        else:
            # Fallback to ratio + line snapping if the grid/OCR anchors are missing.
            h_lines, v_lines = _extract_table_lines(normalized, is_webcam=is_webcam)
            left, right, top, bottom = _calibrated_question_rect(
                question_name,
                normalized.shape[1],
                normalized.shape[0],
                h_lines,
                v_lines,
            )
            crop_img, rect = _crop_with_padding(
                normalized,
                top,
                bottom,
                left,
                right,
                0.03,
                0.15,
            )
            x, y, w, h = rect

        mark_pred = predict_mark_combined(crop_img, q_max) if crop_img.size > 0 else "-"

        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            debug_img,
            question_name,
            (max(8, x - 100), min(debug_img.shape[0] - 12, y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            debug_img,
            f"={mark_pred}",
            (min(debug_img.shape[1] - 90, x + w + 8), min(debug_img.shape[0] - 12, y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 160, 0) if mark_pred.isdigit() else (0, 0, 200),
            2,
        )
        extracted_data.append({"question": question_name, "mark": mark_pred})

    total = sum(int(item["mark"]) for item in extracted_data if str(item["mark"]).isdigit())

    debug_filename = "debug_" + os.path.basename(img_path)
    debug_path = os.path.join(os.path.dirname(img_path), debug_filename)
    cv2.imwrite(debug_path, debug_img)

    return extracted_data, total, debug_filename


def export_to_excel(roll_number, extracted_data, total, output_path):
    row_data = {"Roll Number": str(roll_number)}
    for item in extracted_data:
        value = item["mark"]
        row_data[item["question"]] = int(value) if str(value).isdigit() else "-"
    row_data["Total"] = int(total)

    df_new = pd.DataFrame([row_data])

    if os.path.exists(output_path):
        df_existing = pd.read_excel(output_path)
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = "-"
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = "-"

        if "Roll Number" in df_existing.columns:
            df_existing["Roll Number"] = df_existing["Roll Number"].astype(str)
            existing_match = df_existing["Roll Number"] == str(roll_number)
            if existing_match.any():  # type: ignore
                for col in df_new.columns:
                    df_existing.loc[existing_match, col] = df_new[col].iloc[0]
                df_combined = df_existing
            else:
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_excel(output_path, index=False)
    return output_path


def export_session_to_excel(session_history, output_path):
    all_data = []
    for record in session_history:
        row_data = {
            "Roll Number": str(record.get("rollNumber", "Unknown")),
            "Date": record.get("date", ""),
        }
        if "results" in record:
            for result in record["results"]:
                question = result.get("question", "")
                mark = result.get("mark", "-")
                row_data[question] = int(mark) if str(mark).isdigit() else "-"
        row_data["Total"] = int(record.get("total", 0))
        all_data.append(row_data)

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_excel(output_path, index=False)
    return output_path
