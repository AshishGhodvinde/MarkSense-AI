import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import easyocr  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from torchvision import transforms  # type: ignore

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
for model_path in ["handwritten_marks_model.pth", "mnist_model.pth"]:
    if not os.path.exists(model_path):
        continue
    try:
        mnist_model.load_state_dict(torch.load(model_path, map_location=device))
        loaded_model = True
        break
    except Exception as model_error:
        print(f"Skipping incompatible model weights from {model_path}: {model_error}")
if not loaded_model:
    print("Proceeding with randomly initialized CNN weights.")
mnist_model.eval()

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

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.15:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area

    if best_quad is not None:
        return four_point_transform(img, best_quad)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = img[y : y + h, x : x + w]
    return cv2.resize(cropped, (CANONICAL_WIDTH, CANONICAL_HEIGHT))


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
        ocr_results = reader.readtext(img, min_size=12, detail=1)
    except Exception:
        return None, None, None

    for bbox, text, prob in ocr_results:
        t = str(text).strip().lower().replace(".", "")
        if not t:
            continue
        cx, cy = _bbox_centroid(bbox)
        y_top, y_bottom = _bbox_y_top_bottom(bbox)

        if ("obt" in t or "0bt" in t) and obt_x_center is None:
            obt_x_center = cx
            obt_y_bottom = y_bottom

        # "Total Mark" often comes as ["Total", "Mark"] or one token depending on OCR.
        if "total" in t:
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 5, 60), 1))
        line_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        projection = line_img.sum(axis=1)
        threshold = projection.max() * 0.35 if projection.size else 0
        return _extract_projection_centers(projection, threshold, 12)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 60)))
    line_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    projection = line_img.sum(axis=0)
    threshold = projection.max() * 0.35 if projection.size else 0
    return _extract_projection_centers(projection, threshold, 12)


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


def _roi_horizontal_lines(binary: np.ndarray, x1: int, x2: int) -> List[int]:
    roi = binary[:, max(0, x1) : min(binary.shape[1], x2)]
    if roi.size == 0:
        return []

    h, w = roi.shape
    kernel_w = max(int(w * 0.75), 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    line_img = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * 0.55:
            continue
        ys.append(int(y + ch / 2))
    return _merge_close_positions(ys, gap=10)


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


def _derive_obt_row_boundaries_from_roi(
    binary: np.ndarray,
    obt_left: int,
    obt_right: int,
    obt_y_bottom: Optional[float],
    total_y_top: Optional[float],
) -> Optional[List[int]]:
    img_h = binary.shape[0]
    y_start = int((obt_y_bottom or (img_h * 0.06)) + 4)
    y_stop = int((total_y_top or (img_h * 0.72)) - 4)
    if y_stop <= y_start:
        return None

    # Detect horizontal separators within the Obt column ROI.
    roi_lines = _roi_horizontal_lines(binary, obt_left, obt_right)
    roi_lines = [y for y in roi_lines if y_start <= y <= y_stop]
    roi_lines = sorted(set(roi_lines))

    seq = _best_11_line_sequence(roi_lines)
    if seq is None:
        # Fallback to global line detection if ROI detection is weak.
        global_lines = _get_line_positions(binary, "horizontal")
        global_lines = [y for y in global_lines if y_start <= y <= y_stop]
        global_lines = _merge_close_positions(global_lines, gap=10)
        seq = _best_11_line_sequence(global_lines)
    return seq


def detect_obt_cells(img: np.ndarray, is_webcam: bool = False) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Detect the 10 Obt. cells (below the printed 'Obt.' header and above the Total row).
    Returns list of 10 rectangles: (x1, y1, x2, y2) in image coordinates.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_webcam:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Vertical lines from full image (more stable for column bounds).
    v_lines = _get_line_positions(binary, "vertical")

    obt_x_center, obt_y_bottom, total_y_top = _ocr_find_obt_and_total_anchors(img)

    obt_left, obt_right = _pick_obt_column_bounds(v_lines, obt_x_center, img.shape[1])
    row_lines = _derive_obt_row_boundaries_from_roi(binary, obt_left, obt_right, obt_y_bottom, total_y_top)
    if row_lines is None or len(row_lines) != 11:
        return None

    # Add small in-cell padding to avoid the grid lines, but keep it tight so we don't crop off digits.
    pad_x = max(3, int((obt_right - obt_left) * 0.02))

    rects: List[Tuple[int, int, int, int]] = []
    for i in range(10):
        y_top = int(row_lines[i])
        y_bottom = int(row_lines[i + 1])
        cell_h = max(1, y_bottom - y_top)
        pad_y = max(2, int(cell_h * 0.12))

        x1 = max(0, obt_left + pad_x)
        x2 = min(img.shape[1], obt_right - pad_x)
        y1 = max(0, y_top + pad_y)
        y2 = min(img.shape[0], y_bottom - pad_y)
        if x2 <= x1 or y2 <= y1:
            continue
        rects.append((x1, y1, x2, y2))

    if len(rects) != 10:
        return None
    return rects


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


def _prepare_digit_mask(crop_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9
    )

    h, w = thresh.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 2, 20), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 20)))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    clean = cv2.subtract(thresh, horizontal_lines)
    clean = cv2.subtract(clean, vertical_lines)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    clean = cv2.dilate(clean, np.ones((2, 2), np.uint8), iterations=1)

    # Remove residual cell borders at the crop edges so grid lines do not become "digits".
    edge_y = max(2, int(h * 0.16))
    edge_x = max(2, int(w * 0.04))
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
    mask = _prepare_digit_mask(crop_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [contour for contour in contours if cv2.contourArea(contour) > 20]
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
        if area < 20 or h < 10 or w < 3:
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

    return [digit for digit in digits if digit.size > 0 and np.count_nonzero(digit) > 10]


def _extract_right_digit_crop(crop_img: np.ndarray) -> Optional[np.ndarray]:
    mask = _prepare_digit_mask(crop_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    width = mask.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 55 or h < 14 or w < 5:
            continue
        if x + w / 2 < width * 0.52:
            continue
        candidate_boxes.append((x, y, w, h))

    if not candidate_boxes:
        return None

    rightmost_x = max(x + w for x, y, w, h in candidate_boxes)
    grouped = []
    for x, y, w, h in candidate_boxes:
        if rightmost_x - (x + w) <= width * 0.16:
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
    if digit_mask.size == 0 or np.count_nonzero(digit_mask) < 40:
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
    right_half = mask[:, int(w * 0.48) :]
    right_ink_ratio = float(np.count_nonzero(right_half)) / float(max(right_half.size, 1))
    if right_ink_ratio < 0.012:
        return False

    contours, _ = cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area >= 70 and ch >= 16 and cw >= 6:
            return True
    return False


def _predict_single_digit(component: np.ndarray) -> Optional[str]:
    if component.size == 0:
        return None

    component_bgr = (
        cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        if len(component.shape) == 2
        else component
    )

    template_pred, template_score = predict_mark_template(component_bgr)
    if template_pred is not None and template_score >= 0.82:
        return template_pred

    cnn_pred, cnn_conf = predict_mark_cnn(component_bgr)
    if cnn_pred.isdigit() and cnn_conf >= 0.92:
        return cnn_pred

    ocr_pred, ocr_conf = predict_mark_easyocr(component_bgr)
    if ocr_pred and ocr_pred.isdigit() and ocr_conf >= 0.55:
        return ocr_pred[-1]
    return None


def predict_mark_combined(crop_img: np.ndarray, max_mark: int) -> str:
    if is_likely_dash(crop_img):
        return "-"

    whole_ocr_pred, whole_ocr_conf = predict_mark_easyocr(crop_img)
    whole_ocr_value = _extract_digits_from_text(whole_ocr_pred, max_mark)
    if whole_ocr_value == "0":
        whole_ocr_value = "-"

    if whole_ocr_value is not None and whole_ocr_value != "-" and whole_ocr_conf >= 0.82:
        return whole_ocr_value

    right_digit_crop = _extract_right_digit_crop(crop_img)
    if right_digit_crop is None:
        if whole_ocr_value is not None and whole_ocr_conf >= 0.45:
            return whole_ocr_value
        return "-"

    if right_digit_crop is not None:
        right_digit = _predict_single_digit(right_digit_crop)
        if right_digit == "0":
            return "-"
        if right_digit is not None and int(right_digit) <= max_mark:
            return right_digit

    digit_images = _segment_digit_images(crop_img)
    digit_values: List[str] = []
    for digit_img in digit_images[:2]:
        digit = _predict_single_digit(digit_img)
        if digit is not None:
            digit_values.append(digit)

    if digit_values:
        numeric = _extract_digits_from_text("".join(digit_values), max_mark)
        if numeric is not None:
            return numeric

    if whole_ocr_value is not None and whole_ocr_conf >= 0.45:
        return whole_ocr_value

    cnn_pred, cnn_conf = predict_mark_cnn(crop_img)
    if cnn_pred.isdigit() and int(cnn_pred) <= max_mark and cnn_conf >= 0.98:
        return cnn_pred

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
    img = cv2.imread(img_path)
    if img is None:
        raise Exception("Failed to load image")

    img = deskew_image(img, is_webcam=is_webcam)
    normalized = locate_marksheet(img)
    normalized = enhance_image(normalized)

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
