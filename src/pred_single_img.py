"""
Quick script for predicting mask on a single image and saving the result.
Usage:
    python src/pred_single_img.py <image_path>
    The predicted mask is saved as IMG_PATH_pred.png
"""

import cv2
import os
import sys
import numpy as np
from feature_extraction import extract_features
from models.transformer_model import get_transformer_inference_model
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IMG_PATH = "~/Pictures/disco_pogo.jpg"
try:
    IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG_PATH
    img_path = os.path.expanduser(IMG_PATH)
    print(f"Loading image from {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Compressing image to 256x256
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
except Exception as e:
    raise KeyError(f"Error loading image: {e}. Check the image path.")
print("Loading pretrained model ...")
model = get_transformer_inference_model()

model_name = "transformer_weighted_best"
model.load_state_dict(
    torch.load(f"trained_models/{model_name}.pth", map_location=DEVICE)
)
model.to(DEVICE)

print("Extracting features from the image.")
try:
    features = extract_features(img_resized)

    if np.isnan(features).any():
        raise ValueError("NaNs found")

    tensor = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    print("Running inference ...")
    with torch.no_grad():
        output = model(tensor)
        prob_mask = torch.sigmoid(output)
        binary_mask = (prob_mask > 0.5).float()
    pred_mask = binary_mask.squeeze().cpu().numpy().astype(np.uint8)

# Predict fully inpainted image if error occurs
except Exception as e:
    print(f"Fallback inference due to error: {e}")
    pred_mask = np.ones((256, 256), dtype=np.uint8)  # fallback prediction

# Prepare predicted mask for contours
mask_uint8 = (pred_mask * 255).astype(np.uint8)
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the inpainted image
contour_vis = img_resized.copy()
cv2.drawContours(contour_vis, contours, -1, (255, 0, 0), thickness=3)  # red contour
combined = np.hstack((img_resized, contour_vis))
# Save visualization

print(f"Saving prediction to {img_path}_pred.png")
out_path = f"{img_path}_pred.png"
cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
