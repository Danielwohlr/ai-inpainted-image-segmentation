"""
Runs inference on the test set and saves the predicted masks as a CSV file.

Also plots user-chosen number of images and originals for visualization (remember to set correct paths)
"""

import os
import numpy as np
import torch
import pandas as pd
import cv2
from tqdm import tqdm
from models.unet_model import get_unet_model
from feature_extraction import extract_features
from models.transformer_model import get_transformer_inference_model
from transform_rle import mask2rle

# Set to True if you want to save CSV with compressed masks
SAVE_CSV = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = "Segformer"
# Load model
if model == "Unet":
    model = get_unet_model().to(DEVICE)
elif model == "Segformer":
    model = get_transformer_inference_model()
else:
    raise ValueError("Invalid model name")

model_name = "transformer_weighted_best"
model.load_state_dict(
    torch.load(f"trained_models/{model_name}.pth", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()


# Paths
test_dir = "dataset/train/train/images"
submission_dir = "submissions"
os.makedirs(submission_dir, exist_ok=True)
# Path for saved predicted masks
os.makedirs("predictions", exist_ok=True)


submission_data = []
image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])

# Inference
for i, fname in enumerate(tqdm(image_files)):
    # Load image
    img_path = os.path.join(test_dir, fname)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    try:
        features = extract_features(img)

        if np.isnan(features).any():
            raise ValueError("NaNs found")

        tensor = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            prob_mask = torch.sigmoid(output)
            binary_mask = (prob_mask > 0.5).float()
        pred_mask = binary_mask.squeeze().cpu().numpy().astype(np.uint8)

    # Predict fully inpainted image if error occurs
    except Exception as e:
        print(f"[SKIPPED] {fname} due to error: {e}")
        pred_mask = np.ones((256, 256), dtype=np.uint8)  # fallback prediction

    # Plot first 10 masks and images
    limit = 13  # Setting to 0 will not plot any images
    if i < limit:
        print(f"Predicting {fname}")
    # Load original (non-inpainted) image for comparison, if they are available
    try:
        original_path = os.path.join("dataset/train/train/originals", fname)
        original_img = cv2.imread(original_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    except:
        print("Continued")
        continue

    # Prepare predicted mask for contours
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours on a copy of the inpainted image
    contour_vis = img.copy()
    cv2.drawContours(contour_vis, contours, -1, (255, 0, 0), thickness=3)  # red contour

    # Make sure all images are the same size (just in case)
    H, W, _ = img.shape
    original_img = cv2.resize(original_img, (W, H))
    contour_vis = cv2.resize(contour_vis, (W, H))
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)
    thickness = 2

    labeled_original = original_img.copy()
    labeled_inpainted = img.copy()
    labeled_mask = contour_vis.copy()

    cv2.putText(
        labeled_original, "Original", (10, 25), font, font_scale, color, thickness
    )
    cv2.putText(
        labeled_inpainted, "Inpainted", (10, 25), font, font_scale, color, thickness
    )
    cv2.putText(
        labeled_mask, "Predicted Mask", (10, 25), font, font_scale, color, thickness
    )

    # Stack: original | inpainted | inpainted with contour
    combined = np.hstack((original_img, img, contour_vis))

    # Save visualization
    out_path = f"predictions/pred_{i}.png"
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if i >= limit:
        break
    rle = mask2rle(pred_mask)
    submission_data.append({"ImageId": fname.replace(".png", ""), "EncodedPixels": rle})

if SAVE_CSV:
    # Save CSV
    df = pd.DataFrame(submission_data)
    df.to_csv(os.path.join(submission_dir, f"submission{model_name}.csv"), index=False)
