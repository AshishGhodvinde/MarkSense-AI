import cv2 # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore

class SimpleMarkDetector:
    def __init__(self, mnist_model, device):
        self.mnist_model = mnist_model
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_mark(self, crop_img):
        """Improved preprocessing for better accuracy"""
        if crop_img.size == 0:
            return np.zeros((28, 28), dtype=np.uint8)
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding for varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Remove border noise
        h, w = adaptive_thresh.shape
        border = max(1, min(h, w) // 8)
        adaptive_thresh[:border, :] = 0
        adaptive_thresh[-border:, :] = 0
        adaptive_thresh[:, :border] = 0
        adaptive_thresh[:, -border:] = 0
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find bounding box
        coords = cv2.findNonZero(adaptive_thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            if w > 3 and h > 3:  # Minimum size check
                # Additional check for aspect ratio
                aspect_ratio = w / h
                if 0.3 <= aspect_ratio <= 3.0:  # Reasonable digit aspect ratio
                    adaptive_thresh = adaptive_thresh[y:y+h, x:x+w]
        
        # Resize to 28x28
        resized = cv2.resize(adaptive_thresh, (28, 28), interpolation=cv2.INTER_AREA)
        return resized
    
    def predict_mark(self, processed_img):
        """Improved prediction with better confidence checking"""
        if np.sum(processed_img) == 0:
            return '-', 0.0
        
        img_pil = Image.fromarray(processed_img)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.mnist_model(tensor)
            probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
            
            # Get top 2 predictions
            top2_probs, top2_preds = torch.topk(probabilities, 2)
            
            max_prob = top2_probs[0][0].item()
            second_max_prob = top2_probs[0][1].item()
            prediction = str(top2_preds[0][0].item())
            
            # More strict confidence checking
            if max_prob > 0.7 and (max_prob - second_max_prob) > 0.3:
                return prediction, max_prob
            elif max_prob > 0.8:  # Very high confidence
                return prediction, max_prob
            else:
                return '-', max_prob
