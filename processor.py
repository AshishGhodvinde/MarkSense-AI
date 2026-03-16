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
from typing import Optional, Dict, Any, List

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

print("Initializing MNIST Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mnist_model = MnistCNN().to(device)

if os.path.exists('mnist_model.pth'):
    mnist_model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
mnist_model.eval()

def preprocess_for_mnist(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Remove horizontal lines using morphology (only lines spanning >85% of width)
    h, w = thresh.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.85), 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)
        
    # Remove vertical lines (only lines spanning >85% of height)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.85)))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((28, 28), dtype=np.uint8)
        
    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    
    # Filter out tiny specks of noise that act as a dash, or actual dashes (-)
    if cw < 3 or ch < 3:
        return np.zeros((28, 28), dtype=np.uint8)
        
    # If it is much wider than it is tall, it's a dash '-', not a digit!
    if cw > 1.8 * ch:
        return np.zeros((28, 28), dtype=np.uint8)
        
    digit_thresh = thresh[y:y+ch, x:x+cw]
    height, width = digit_thresh.shape
    max_dim = max(height, width)
        
    canvas = np.zeros((max_dim + 12, max_dim + 12), dtype=np.uint8)
    y_off = (canvas.shape[0] - height) // 2
    x_off = (canvas.shape[1] - width) // 2
    canvas[y_off:y_off+height, x_off:x_off+width] = digit_thresh
    
    canvas = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Save for debugging
    import uuid
    os.makedirs('debug_digits', exist_ok=True)
    cv2.imwrite(f'debug_digits/{uuid.uuid4().hex}.jpeg', canvas)
    
    return canvas

def predict_mark(crop_img):
    img_processed = preprocess_for_mnist(crop_img)
    if np.sum(img_processed) == 0:
        return '-'
        
    img_pil = Image.fromarray(img_processed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = mnist_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        
        if max_prob.item() < 0.3:
            return '-'
            
    return str(predicted.item())

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise Exception("Failed to load image")
    try:
        # Pass the pre-loaded OpenCV img array directly to avoid EXIF mismatch
        # min_size prevents the internal EasyOCR bug where tiny 1-pixel contours cause cv2.resize to crash on dimension 0
        results = reader.readtext(img, min_size=10)
    except Exception as e:
        print(f"EasyOCR crash bypassed. Attempting rescale. Error: {e}")
        # Fallback: slightly blur and rescale to destroy the microscopic anomalous contours crashing CRAFT
        safe_img = cv2.GaussianBlur(img.copy(), (3, 3), 0)
        results = reader.readtext(safe_img, min_size=20)
    ocr_elements = []
    
    img_debug = img.copy()

    for (bbox, text, prob) in results:
        y_center = sum([p[1] for p in bbox]) / 4
        x_center = sum([p[0] for p in bbox]) / 4
        h = abs(bbox[0][1] - bbox[2][1])
        w = abs(bbox[0][0] - bbox[2][0])
        ocr_elements.append({
            'text': str(text).strip(), 
            'x': float(x_center), 
            'y': float(y_center),
            'h': float(h),
            'w': float(w),
            'bbox': bbox
        })
        cv2.rectangle(img_debug, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (255, 0, 0), 1)

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

    obt_x: float = float(obt_e['x']) if obt_e is not None else float(img_width) * 0.75 # type: ignore
    max_x: float = float(max_e['x']) if max_e is not None else float(img_width) * 0.5 # type: ignore

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

    expected_q = [
        ('Q1.a', 2), ('Q1.b', 2), ('Q1.c', 2), ('Q1.d', 2), ('Q1.e', 2), ('Q1.f', 2),
        ('Q2.a', 5), ('Q2.b', 5),
        ('Q3.a', 5), ('Q3.b', 5)
    ]
    extracted_data = []
    
    q_index = 0
    for r in rows:
        if q_index >= len(expected_q): break
        
        texts = [str(e['text']) for e in r['elems']]
        full_text = " ".join(texts).lower()
        
        # Completely skip any rows that are on or below the "Total" row
        if float(r['avg_y']) > float(total_y) - 15.0:
            continue
        
        if 'max' in full_text or 'obt' in full_text or 'total' in full_text or 'sign' in full_text or 'eval' in full_text:
            continue
            
        # If there's any text in this row that sits near the 'Max' column, it's a valid row
        valid_row = False
        for e in r['elems']:
            ex = float(e['x'])
            if abs(ex - float(max_x)) < float(img_width) * 0.15:
                valid_row = True
                break
                
        if valid_row:
            question_name, q_max = expected_q[q_index] # type: ignore

    
            
            crop_w = int(float(img_width) * 0.15)
            # Reduce height and shift Y heavily UP (so it sits on the line instead of crossing it)
            crop_h = int(med_h * 1.5)
            c_x = int(obt_x - crop_w / 2)
            c_y = int(float(r['avg_y']) - med_h * 0.9)
            
            c_x = max(0, min(c_x, img_width - 1))
            c_y = max(0, min(c_y, img_height - 1))
            x_end = min(c_x + crop_w, img_width)
            y_end = min(c_y + crop_h, img_height)
            
            cv2.rectangle(img_debug, (c_x, c_y), (x_end, y_end), (0, 0, 255), 2)
            cv2.putText(img_debug, question_name, (c_x-60, c_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            crop_img = img[c_y:y_end, c_x:x_end] # type: ignore
            mark_pred = '-'
            
            if crop_img.size > 0:
                obt_candidates = [e for e in r['elems'] if c_x < e['x'] < x_end]
                easyocr_mark = None
                if obt_candidates:
                    best_cand = min(obt_candidates, key=lambda e: abs(e['x'] - obt_x))
                    t = best_cand['text'].replace('O', '0').replace('l', '1').replace('I', '1').strip()
                    if t.isdigit():
                        easyocr_mark = t
                
                cnn_pred = predict_mark(crop_img)
                
                if cnn_pred.isdigit():
                    mark_pred = cnn_pred
                elif easyocr_mark is not None:
                    mark_pred = easyocr_mark
                    
                if mark_pred.isdigit():
                    val = int(mark_pred)
                    if val > q_max:
                        if val == 7: mark_pred = '1'
                        elif val == 9: mark_pred = '4'
                        else: mark_pred = '-'
            
            extracted_data.append({
                "question": question_name,
                "mark": mark_pred
            })
            q_index += 1

    # Fill remaining questions if any were missed
    while q_index < len(expected_q):
        extracted_data.append({
            "question": expected_q[q_index][0],
            "mark": "-"
        })
        q_index += 1

    total = sum([int(m['mark']) for m in extracted_data if str(m['mark']).isdigit()])
    
    debug_filename = "debug_" + os.path.basename(img_path)
    debug_path = os.path.join(os.path.dirname(img_path), debug_filename)
    cv2.imwrite(debug_path, img_debug)
    
    return extracted_data, total, debug_filename

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
