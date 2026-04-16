"""Diagnostic: see what EasyOCR detects on the full image to understand what data is available."""
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

for img_path in ['images/1.jpeg', 'images/16.jpeg']:
    print(f"\n=== OCR results for {img_path} ===")
    img = cv2.imread(img_path)
    results = reader.readtext(img, min_size=10)
    
    # Sort by Y position
    results.sort(key=lambda r: sum(p[1] for p in r[0]) / 4)
    
    for bbox, text, conf in results:
        y_center = sum(p[1] for p in bbox) / 4
        x_center = sum(p[0] for p in bbox) / 4
        print(f"  x={x_center:6.1f} y={y_center:6.1f} conf={conf:.3f} text='{text}'")
