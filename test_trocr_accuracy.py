import os
os.environ["USE_TF"] = "0"
import warnings
warnings.filterwarnings("ignore")
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import numpy as np
from processor import detect_table_cells, deskew_image, enhance_image, is_likely_dash

print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def predict_mark_trocr(crop_img: np.ndarray) -> str:
    if is_likely_dash(crop_img):
        return "-"
        
    image = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=5)
        
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Simple post-processing
    text = text.replace('O', '0').replace('l', '1').replace('I', '1').replace('S', '5')
    text = text.replace('o', '0').replace('s', '5').replace('Z', '2').replace('z', '2')
    
    # Just grab any digit
    for ch in text:
        if ch.isdigit():
            return ch
    return "-"

def get_crops_for_testing(img_path):
    img = cv2.imread(img_path)
    img = deskew_image(img)
    img_enhanced = enhance_image(img)
    
    h_lines, max_col_x, obt_col_x = detect_table_cells(img_enhanced)
    
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    ocr_results = reader.readtext(img_enhanced, min_size=10)
    
    ocr_elements = []
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
        
    obt_e = None
    for e in ocr_elements:
        t = str(e['text']).lower().replace('.', '').strip()
        if ('obt' in t or '0bt' in t) and obt_e is None:
            obt_e = e

    img_width = img.shape[1]
    img_height = img.shape[0]

    if obt_e is not None:
        obt_x = float(obt_e['x'])
    elif obt_col_x is not None:
        obt_x = float(obt_col_x) + (img_width - obt_col_x) / 2.0
    else:
        obt_x = float(img_width) * 0.82

    rows = []
    med_h = float(np.median([float(e['h']) for e in ocr_elements if float(e['h']) > 10])) if ocr_elements else 40.0
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
        'Q1.a', 'Q1.b', 'Q1.c', 'Q1.d', 'Q1.e', 'Q1.f',
        'Q2.a', 'Q2.b',
        'Q3.a', 'Q3.b'
    ]
    
    crops = {}
    q_index = 0
    
    # very simplified logic just to grab crops
    for r in rows:
        if q_index >= len(expected_q): break
        valid_row = False
        for e in r['elems']:
            if abs(float(e['x']) - (img_width * 0.65)) < img_width * 0.15:
                valid_row = True
                break
        if not valid_row: continue
        
        crop_w = int(float(img_width) * 0.13)
        crop_h = int(med_h * 1.4)
        c_x = int(obt_x - crop_w / 2)
        c_y = int(float(r['avg_y']) - med_h * 0.8)
        
        c_x = max(0, min(c_x, img_width - 1))
        c_y = max(0, min(c_y, img_height - 1))
        x_end = min(c_x + crop_w, img_width)
        y_end = min(c_y + crop_h, img_height)

        crop_img = img_enhanced[c_y:y_end, c_x:x_end]
        crops[expected_q[q_index]] = crop_img
        q_index += 1
        
    return crops

gt_16 = {
    'Q1.a': '2', 'Q1.b': '-', 'Q1.c': '2', 'Q1.d': '1',
    'Q1.e': '2', 'Q1.f': '-', 'Q2.a': '-', 'Q2.b': '5',
    'Q3.a': '4', 'Q3.b': '-'
}

crops = get_crops_for_testing('images/16.jpeg')
for q, img in crops.items():
    res = predict_mark_trocr(img)
    expected = gt_16.get(q, '?')
    print(f"{q}: expected {expected}, TrOCR predicted {res}")
