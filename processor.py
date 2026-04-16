import cv2 # type: ignore
import numpy as np # type: ignore
import easyocr # type: ignore
import pandas as pd # type: ignore
import os
import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
from train import MnistCNN # type: ignore
import re
import warnings
from typing import Optional, Dict, Any, List, Tuple

# Suppress the noisy pin_memory warnings from PyTorch DataLoader
warnings.filterwarnings("ignore", message=".*pin_memory.*")

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

print("Initializing MNIST Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mnist_model = MnistCNN().to(device)

if os.path.exists('mnist_model.pth'):
    mnist_model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
mnist_model.eval()

# ─── PREPROCESSING UTILITIES ──────────────────────────────────────────────────

def enhance_image(img: np.ndarray) -> np.ndarray:
    """Apply contrast enhancement and denoising to improve OCR and digit recognition."""
    # Convert to LAB color space for better contrast manipulation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Light denoising to remove scanner/camera noise without blurring digits
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 6, 6, 7, 21)
    return denoised


def deskew_image(img: np.ndarray, is_webcam: bool = False) -> np.ndarray:
    """Detect and correct slight rotation in scanned images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line = 50 if is_webcam else 100
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=min_line, maxLineGap=10)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (table grid lines)
        if abs(angle) < 15 if is_webcam else 10:
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:
        return img  # Already straight enough

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ─── TABLE STRUCTURE DETECTION ────────────────────────────────────────────────

def detect_table_cells(img: np.ndarray, is_webcam: bool = False) -> Tuple[List[List[int]], Optional[int], Optional[int]]:
    """
    Detect the table grid in the marksheet and identify cell positions.
    Returns: (list of horizontal line y-positions, max_col_x, obt_col_x)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if is_webcam:
        # Webcams have glare, Otsu fails. Use robust adaptive thresholding.
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 10)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h, w = binary.shape

    # Detect horizontal lines
    h_len = max(w // 5, 30) if is_webcam else max(w // 3, 50)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    v_len = max(h // 10, 20) if is_webcam else max(h // 6, 30)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Find horizontal line y-positions
    h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_lines_y = sorted(set([int(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2) for c in h_contours]))

    # Merge lines that are very close together (within 10px)
    merged_h_lines = []
    for y in h_lines_y:
        if not merged_h_lines or abs(y - merged_h_lines[-1]) > 10:
            merged_h_lines.append(y)

    # Find vertical line x-positions
    v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_lines_x = sorted(set([int(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2) for c in v_contours]))

    # Merge close vertical lines
    merged_v_lines = []
    for x in v_lines_x:
        if not merged_v_lines or abs(x - merged_v_lines[-1]) > 10:
            merged_v_lines.append(x)

    # The last two vertical lines should be Max and Obt columns
    max_col_x = None
    obt_col_x = None
    if len(merged_v_lines) >= 3:
        # Second-to-last gap = Max column, last gap = Obt column
        obt_col_x = merged_v_lines[-2]  # Left edge of Obt column
        max_col_x = merged_v_lines[-3] if len(merged_v_lines) >= 4 else merged_v_lines[-2]

    return merged_h_lines, max_col_x, obt_col_x


def extract_obt_cells(img: np.ndarray, h_lines: List[int],
                       obt_col_x: Optional[int]) -> List[Tuple[np.ndarray, int, int, int, int]]:
    """
    Extract individual cells from the Obt. column using detected grid lines.
    Returns list of (cell_image, x, y, w, h) tuples.
    """
    _, w_img = img.shape[:2]
    cells = []

    if obt_col_x is None:
        return cells

    # The Obt column goes from obt_col_x to the right edge (or next vertical line)
    col_left = obt_col_x
    col_right = w_img  # Use full width to the right

    for i in range(len(h_lines) - 1):
        y_top = h_lines[i]
        y_bot = h_lines[i + 1]

        cell_height = y_bot - y_top
        if cell_height < 15:
            continue  # Skip tiny gaps

        # Add padding inside the cell to avoid grid lines
        pad_y = max(3, int(cell_height * 0.1))
        pad_x = 5

        cy1 = y_top + pad_y
        cy2 = y_bot - pad_y
        cx1 = col_left + pad_x
        cx2 = col_right - pad_x

        if cy2 <= cy1 or cx2 <= cx1:
            continue

        cell_img = img[cy1:cy2, cx1:cx2]
        if cell_img.size > 0:
            cells.append((cell_img, cx1, cy1, cx2 - cx1, cy2 - cy1))

    return cells


# ─── DIGIT RECOGNITION ────────────────────────────────────────────────────────

def is_likely_dash(crop_img: np.ndarray) -> bool:
    """
    Determine if a crop contains a dash '-' rather than a digit.
    Uses aspect ratio analysis and ink density.
    """
    if crop_img.size == 0:
        return True

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Check total ink density — very low means empty or just a tiny mark
    ink_ratio = np.sum(binary > 0) / binary.size
    if ink_ratio < 0.02:
        return True  # Nearly empty cell

    # Find contours and analyze the dominant shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 15]
    if not valid:
        return True

    c = max(valid, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)

    # Strong dash detection: much wider than tall
    if ch < 6:
        return True
    if cw > 1.8 * ch and ch < 15:
        return True

    return False


def clean_crop_for_ocr(crop_img: np.ndarray) -> np.ndarray:
    """
    Remove grid lines and background noise from a crop, returning an 
    image suitable for EasyOCR without resizing to tiny dimensions.
    """
    if crop_img.size == 0:
        return crop_img

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img.copy()

    # Adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    h, w = thresh.shape

    # Remove horizontal lines
    if w > 20:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 2, 15), 1))
        h_detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
        thresh = cv2.subtract(thresh, h_detected)

    # Remove vertical lines
    if h > 20:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 15)))
        v_detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)
        thresh = cv2.subtract(thresh, v_detected)

    # Clean up small noise
    clean_kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    # Slightly dilate to connect strokes for OCR
    dilate_kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, dilate_kernel, iterations=1)
    
    # Invert back to black text on white background (EasyOCR prefers this)
    clean_img = cv2.bitwise_not(thresh)

    # Convert back to BGR for EasyOCR compatibility
    clean_img_bgr = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)
    
    # Add a border to give EasyOCR some padding
    clean_padded = cv2.copyMakeBorder(clean_img_bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return clean_padded


def preprocess_for_mnist(crop_img: np.ndarray) -> np.ndarray:
    """
    Preprocess a cell crop for MNIST-style CNN input.
    Focus on isolating the handwritten digit from grid lines and noise.
    """
    if crop_img.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img.copy()

    # Use adaptive thresholding — handles varying ink density and backgrounds
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    h, w = thresh.shape

    # Remove horizontal lines (grid artifacts) — use relative kernel size
    if w > 20:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 2, 15), 1))
        h_detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
        thresh = cv2.subtract(thresh, h_detected)

    # Remove vertical lines (grid artifacts)
    if h > 20:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 15)))
        v_detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)
        thresh = cv2.subtract(thresh, v_detected)

    # Clean up with morphological opening
    clean_kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    # Find the main digit contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((28, 28), dtype=np.uint8)

    # Filter out tiny noise contours and keep the largest meaningful one
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not valid_contours:
        return np.zeros((28, 28), dtype=np.uint8)

    c = max(valid_contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)

    # Filter out dashes (much wider than tall) and tiny specks
    if cw < 4 or ch < 4:
        return np.zeros((28, 28), dtype=np.uint8)

    if cw > 2.0 * ch:
        return np.zeros((28, 28), dtype=np.uint8)

    digit_crop = thresh[y:y + ch, x:x + cw]

    # Center the digit on a square canvas with padding (MNIST-style)
    max_dim = max(ch, cw)
    padding = max(4, max_dim // 4)
    canvas_size = max_dim + 2 * padding
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    y_off = (canvas_size - ch) // 2
    x_off = (canvas_size - cw) // 2
    canvas[y_off:y_off + ch, x_off:x_off + cw] = digit_crop

    # Resize to 28x28
    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

    # Apply slight dilation to thicken thin strokes (matches MNIST stroke width better)
    dilate_kernel = np.ones((2, 2), np.uint8)
    resized = cv2.dilate(resized, dilate_kernel, iterations=1)

    return resized


def predict_mark_cnn(crop_img: np.ndarray) -> Tuple[str, float]:
    """
    Predict digit using CNN model.
    Returns (predicted_digit_string, confidence).
    """
    img_processed = preprocess_for_mnist(crop_img)
    if np.sum(img_processed) == 0:
        return '-', 0.0

    img_pil = Image.fromarray(img_processed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = mnist_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs.data, dim=1)

        # Get top-3 predictions
        top_probs, top_preds = torch.topk(probabilities, min(3, probabilities.shape[1]))
        max_prob = top_probs[0][0].item()
        prediction = str(top_preds[0][0].item())

    return prediction, max_prob


def predict_mark_easyocr(crop_img: np.ndarray) -> Tuple[str, float]:
    """
    Predict digit using EasyOCR.
    Returns (predicted_digit_string, confidence).
    """
    try:
        # Scale up image for better OCR small digit recognition
        h, w = crop_img.shape[:2]
        if min(h, w) > 0:
            scale = 2.0
            enlarged = cv2.resize(crop_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            enlarged = crop_img
            
        # Try with digit-only allowlist first
        results = reader.readtext(enlarged, allowlist='0123456789-', min_size=5)
        if results:
            # Take the result with highest confidence
            best = max(results, key=lambda r: r[2])
            text = best[1].strip()
            conf = best[2]
            # If OCR reads a dash explicitly
            if text == '-' or text == '—' or text == '–':
                return '-', conf
            # Clean up common OCR misreads
            text = text.replace('O', '0').replace('l', '1').replace('I', '1').replace('S', '5')
            text = text.replace('o', '0').replace('s', '5').replace('Z', '2').replace('z', '2')
            # Extract first digit if multi-char
            for ch in text:
                if ch.isdigit():
                    return ch, conf
        
        # Also try without allowlist (sometimes allowlist hurts accuracy)
        results2 = reader.readtext(enlarged, min_size=5)
        if results2:
            best = max(results2, key=lambda r: r[2])
            text = best[1].strip()
            conf = best[2]
            
            # Additional cleanup for marks like "1/" or "/2"
            text = text.replace('/', '1').replace('|', '1').replace('!', '1').replace(']', '1')
            text = text.replace('O', '0').replace('l', '1').replace('I', '1').replace('S', '5')
            text = text.replace('o', '0').replace('s', '5').replace('Z', '2').replace('z', '2')
            
            # Special case for 1 which is often misread as a tall line
            if len(text) == 1 and text in ('l', 'I', '|', '/', '\\'):
                return '1', conf * 0.8
                
            for ch in text:
                if ch.isdigit():
                    return ch, conf
    except Exception:
        pass
    return '-', 0.0


def predict_mark_combined(crop_img: np.ndarray, max_mark: int) -> str:
    """
    Combined prediction using CNN + EasyOCR with voting and validation.
    max_mark constrains the valid range (e.g., 2 or 5).
    """
    # Step 0: Dash detection — check if this cell likely contains a dash
    if is_likely_dash(crop_img):
        return '-'

    cnn_pred, cnn_conf = predict_mark_cnn(crop_img)
    
    # Run EasyOCR on BOTH original crop and cleaned crop
    ocr_pred_raw, ocr_conf_raw = predict_mark_easyocr(crop_img)
    
    clean_crop = clean_crop_for_ocr(crop_img)
    ocr_pred_clean, ocr_conf_clean = predict_mark_easyocr(clean_crop)

    # ─── Decision Logic ─────────────────────────────────────────────────

    # Choose best OCR result
    if ocr_conf_clean > ocr_conf_raw:
        ocr_pred, ocr_conf = ocr_pred_clean, ocr_conf_clean
    else:
        ocr_pred, ocr_conf = ocr_pred_raw, ocr_conf_raw

    # If EasyOCR returns a dash with any confidence, trust it
    if ocr_pred == '-':
        if cnn_pred == '-' or ocr_conf > 0.4:
            return '-'

    candidates = []

    # Add OCR prediction (prefer OCR — it's more reliable for real handwriting)
    if ocr_pred.isdigit() and ocr_conf >= 0.15:
        candidates.append((ocr_pred, ocr_conf + 0.3, 'ocr'))  # Hefty boost for OCR

    # Add CNN prediction if confident enough
    if cnn_pred.isdigit() and cnn_conf >= 0.8:
        candidates.append((cnn_pred, cnn_conf, 'cnn'))

    if not candidates:
        # Both methods failed — likely a dash or empty cell
        return '-'

    # If both agree, very high confidence — use it
    if len(candidates) == 2 and candidates[0][0] == candidates[1][0]:
        digit = int(candidates[0][0])
        if digit <= max_mark:
            return str(digit)
        else:
            return '-'

    # Filter to valid range only
    valid_candidates = [(p, c, s) for p, c, s in candidates if int(p) <= max_mark]

    if valid_candidates:
        # Sort by confidence, preferring OCR in ties
        valid_candidates.sort(key=lambda x: (x[1], x[2] == 'ocr'), reverse=True)
        return valid_candidates[0][0]

    # If no valid candidate (both out of range), try to map common misreads
    for pred, conf, source in candidates:
        digit = int(pred)
        if digit == 7:
            return '1'  # 7 is often misread 1
        elif digit == 9:
            return '4'  # 9 is often misread 4
        elif digit == 8 and max_mark >= 5:
            return '5'  # 8 can look like 5 in handwriting
        elif digit == 6 and max_mark >= 5:
            return '5'

    return '-'


# ─── MAIN PROCESSING PIPELINE ────────────────────────────────────────────────

def process_image(img_path: str, session_folder: str = None, is_webcam: bool = False) -> Tuple[List[Dict], int, str]:
    """
    Main processing pipeline for a single marksheet image.
    Uses structural table detection + multi-method digit recognition.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise Exception("Failed to load image")

    # Step 1: Enhance and deskew
    img = deskew_image(img, is_webcam=is_webcam)
    img_enhanced = enhance_image(img)

    # Step 2: Detect table structure
    h_lines, max_col_x, obt_col_x = detect_table_cells(img_enhanced, is_webcam=is_webcam)

    # Step 3: OCR pass for structural labels (Q1, Q2, Max, Obt, Total, etc.)
    try:
        ocr_results = reader.readtext(img_enhanced, min_size=10)
    except Exception as e:
        print(f"EasyOCR crash bypassed. Attempting rescale. Error: {e}")
        safe_img = cv2.GaussianBlur(img_enhanced.copy(), (3, 3), 0)
        ocr_results = reader.readtext(safe_img, min_size=20)

    ocr_elements = []
    img_debug = img.copy()

    for (bbox, text, prob) in ocr_results:
        y_center = sum([p[1] for p in bbox]) / 4
        x_center = sum([p[0] for p in bbox]) / 4
        h_el = abs(bbox[0][1] - bbox[2][1])
        w_el = abs(bbox[0][0] - bbox[2][0])
        ocr_elements.append({
            'text': str(text).strip(),
            'x': float(x_center),
            'y': float(y_center),
            'h': float(h_el),
            'w': float(w_el),
            'bbox': bbox
        })
        cv2.rectangle(img_debug, (int(bbox[0][0]), int(bbox[0][1])),
                      (int(bbox[2][0]), int(bbox[2][1])), (255, 0, 0), 1)

    # Find key structural elements
    max_e: Optional[Dict[str, Any]] = None
    obt_e: Optional[Dict[str, Any]] = None
    img_height = int(img.shape[0])
    img_width = int(img.shape[1])
    total_y = float(img_height)

    ocr_elements.sort(key=lambda x: x['y'])

    for e in ocr_elements:
        t = str(e['text']).lower().replace('.', '').strip()
        if 'max' in t and max_e is None:
            max_e = e
        if ('obt' in t or '0bt' in t) and obt_e is None:
            obt_e = e
        alpha = re.sub(r'[^a-z]', '', t)
        if 'total' in alpha:
            total_y = float(e['y'])

    # Determine column positions — prioritize the OCR text headers for perfect center alignment
    if obt_e is not None:
        obt_x = float(obt_e['x'])
    elif obt_col_x is not None:
        obt_x = float(obt_col_x) + (img_width - obt_col_x) / 2.0
    else:
        obt_x = float(img_width) * 0.82

    if max_e is not None:
        max_x_pos = float(max_e['x'])
    elif max_col_x is not None:
        max_x_pos = float(max_col_x) + (obt_x - max_col_x) / 2.0 if obt_x else float(max_col_x)
    else:
        max_x_pos = float(img_width) * 0.65

    # Step 4: Build rows from OCR elements
    rows: List[Dict[str, Any]] = []
    med_h: float = float(np.median([float(e['h']) for e in ocr_elements if float(e['h']) > 10])) if ocr_elements else 40.0

    for e in ocr_elements:
        placed = False
        for r in rows:
            if abs(float(e['y']) - float(r['avg_y'])) < med_h * 0.7:
                r['elems'].append(e)
                r['avg_y'] = sum(float(el['y']) for el in r['elems']) / len(r['elems'])
                placed = True
                break
        if not placed:
            rows.append({'avg_y': e['y'], 'elems': [e]})

    rows.sort(key=lambda x: x['avg_y'])

    # Step 5: Extract marks for each question
    expected_q = [
        ('Q1.a', 2), ('Q1.b', 2), ('Q1.c', 2), ('Q1.d', 2), ('Q1.e', 2), ('Q1.f', 2),
        ('Q2.a', 5), ('Q2.b', 5),
        ('Q3.a', 5), ('Q3.b', 5)
    ]
    extracted_data = []
    q_index = 0

    for r in rows:
        if q_index >= len(expected_q):
            break

        texts = [str(e['text']) for e in r['elems']]
        full_text = " ".join(texts).lower()

        # Skip rows on or below the "Total" row
        if float(r['avg_y']) > float(total_y) - 15.0:
            continue

        # Skip header and footer rows
        if any(kw in full_text for kw in ['max', 'obt', 'total', 'sign', 'eval', 'student', 'teacher']):
            continue

        # Validate this is a data row (contains text near Max column)
        valid_row = False
        for e in r['elems']:
            ex = float(e['x'])
            if abs(ex - float(max_x_pos)) < float(img_width) * 0.15:
                valid_row = True
                break

        if valid_row:
            question_name, q_max = expected_q[q_index]

            # Calculate crop region — centered on the Obt column for this row
            crop_w = int(float(img_width) * 0.13)
            crop_h = int(med_h * 1.4)
            c_x = int(obt_x - crop_w / 2)
            c_y = int(float(r['avg_y']) - med_h * 0.8)

            # Clamp to image bounds
            c_x = max(0, min(c_x, img_width - 1))
            c_y = max(0, min(c_y, img_height - 1))
            x_end = min(c_x + crop_w, img_width)
            y_end = min(c_y + crop_h, img_height)

            # Draw debug rectangle
            cv2.rectangle(img_debug, (c_x, c_y), (x_end, y_end), (0, 0, 255), 2)
            cv2.putText(img_debug, question_name, (c_x - 60, c_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            mark_pred = '-'
            
            # STEP 5.1: Check if EasyOCR full-image scan already found a digit perfectly in this cell
            best_ocr_elem = None
            for e in r['elems']:
                ex = float(e['x'])
                # If element is horizontally positioned within the "Obt" column cell bounds
                if abs(ex - obt_x) < crop_w * 0.6:
                    t = e['text'].strip()
                    if t == '-' or t == '—' or t == '–':
                        best_ocr_elem = '-'
                        break
                    
                    # Clean up common misreads
                    t = t.replace('/', '1').replace('|', '1').replace('!', '1').replace(']', '1')
                    t = t.replace('O', '0').replace('l', '1').replace('I', '1').replace('S', '5')
                    t = t.replace('o', '0').replace('s', '5').replace('Z', '2').replace('z', '2')
                    
                    if len(t) == 1 and t in ('l', 'I', '|', '/', '\\'):
                        best_ocr_elem = '1'
                        break
                        
                    for ch in t:
                        if ch.isdigit():
                            digit = int(ch)
                            if digit <= q_max:
                                best_ocr_elem = str(digit)
                                break
                    if best_ocr_elem:
                        break

            # STEP 5.2: Fallback to cropping and CNN / Local OCR if no full-image match was found
            if best_ocr_elem is not None:
                mark_pred = best_ocr_elem
            else:
                crop_img = img_enhanced[c_y:y_end, c_x:x_end]
                if crop_img.size > 0:
                    # Use combined multi-method prediction
                    mark_pred = predict_mark_combined(crop_img, q_max)

            # Draw prediction on debug image
            color = (0, 180, 0) if mark_pred.isdigit() else (0, 0, 200)
            cv2.putText(img_debug, f"={mark_pred}", (x_end + 5, c_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            extracted_data.append({
                "question": question_name,
                "mark": mark_pred
            })
            q_index += 1

    # Fill remaining questions if any rows were missed
    while q_index < len(expected_q):
        extracted_data.append({
            "question": expected_q[q_index][0],
            "mark": "-"
        })
        q_index += 1

    total = sum([int(m['mark']) for m in extracted_data if str(m['mark']).isdigit()])

    # Save debug image
    debug_filename = "debug_" + os.path.basename(img_path)
    debug_path = os.path.join(os.path.dirname(img_path), debug_filename)
    cv2.imwrite(debug_path, img_debug)

    return extracted_data, total, debug_filename


# ─── EXCEL EXPORT ─────────────────────────────────────────────────────────────

def export_to_excel(roll_number, extracted_data, total, output_path):
    row_data = {"Roll Number": roll_number}
    for item in extracted_data:
        row_data[item['question']] = item['mark']
    row_data["Total"] = total

    df_new = pd.DataFrame([row_data])

    if os.path.exists(output_path):
        df_existing = pd.read_excel(output_path)
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = "-"
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = "-"

        # If the roll number already exists, update that specific row instead of blindly appending
        # We enforce type string for safer matching
        if 'Roll Number' in df_existing.columns:
            df_existing['Roll Number'] = df_existing['Roll Number'].astype(str)
            existing_match = df_existing['Roll Number'] == str(roll_number)
            if existing_match.any(): # type: ignore
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
    """Export all session data to a single Excel file"""
    all_data = []

    for record in session_history:
        roll_number = record.get('rollNumber', 'Unknown')
        total = record.get('total', 0)
        date = record.get('date', '')

        # Create a row for each student - start with basic info
        row_data = {
            'Roll Number': roll_number,
            'Date': date
        }

        # Add individual marks if available (before total)
        if 'results' in record:
            for result in record['results']:
                question = result.get('question', '')
                mark = result.get('mark', 0)
                row_data[f'Q_{question}'] = mark

        # Add Total Score at the end
        row_data['Total Score'] = total

        all_data.append(row_data)

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_excel(output_path, index=False)
    return output_path
