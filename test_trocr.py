import os
os.environ["USE_TF"] = "0"
import warnings
warnings.filterwarnings("ignore")
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

print("Loading TrOCR model (this may take a minute)...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def ocr_trocr(img_path):
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=5)
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

print("TrOCR loaded successfully.")
