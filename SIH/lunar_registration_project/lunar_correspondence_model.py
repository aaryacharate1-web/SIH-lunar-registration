"""SIH 2026 26166: lunar cross-sensor correspondence and registration.

Put a reference image and a moving image in data/, then set their filenames
in the CONFIGURATION section below. This project never assumes that the
official SIH dataset has a public download link.
"""

from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.feature import LoFTR


# ----------------------- CONFIGURATION -----------------------
PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = PROJECT_DIR / "data" / "lro_nac_reference.png"  # TODO: change
MOVING_IMAGE = PROJECT_DIR / "data" / "chandrayaan_ohrc_image.png"  # TODO: change
TRANSFORM_TYPE = "affine"  # "affine" is best for a local overlap; try "homography" too.
CONFIDENCE_THRESHOLD = 0.70
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def normalize_lunar_image(image: np.ndarray) -> np.ndarray:
    """Percentile stretch + CLAHE, intended for one selected science band."""
    image = image.astype(np.float32)
    valid = np.isfinite(image)
    if valid.sum() == 0:
        raise ValueError("Image has no valid pixels.")
    low, high = np.percentile(image[valid], [1, 99])
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    image[~valid] = 0
    image = (image * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def load_grayscale(path: Path) -> np.ndarray:
    """Loads a common 8/16-bit grayscale image. Select an IIRS band before use."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                   nn.BatchNorm2d(out_channels))
                     if stride != 1 or in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + residual)


class LunarCorrespondenceNet(nn.Module):
    """Siamese dense descriptor network; run it on both image inputs."""
    def __init__(self, descriptor_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32, 64, 2), ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2), ResidualBlock(128, 128), ResidualBlock(128, 128),
        )
        self.head = nn.Sequential(nn.Conv2d(128, descriptor_dim, 1), nn.BatchNorm2d(descriptor_dim))

    def forward(self, image):
        return F.normalize(self.head(self.encoder(image)), p=2, dim=1)


@torch.no_grad()
def find_matches(model, reference: np.ndarray, moving: np.ndarray):
    """Pretrained LoFTR dense matches in original image pixel coordinates."""
    model.eval()
    ref_t = to_tensor(normalize_lunar_image(reference), DEVICE)
    mov_t = to_tensor(normalize_lunar_image(moving), DEVICE)
    matches = model({"image0": ref_t, "image1": mov_t})
    ref_points = matches["keypoints0"].cpu().numpy()
    mov_points = matches["keypoints1"].cpu().numpy()
    scores = matches["confidence"].cpu().numpy()
    keep = scores >= CONFIDENCE_THRESHOLD
    return ref_points[keep], mov_points[keep], scores[keep]


def register_images(model, reference: np.ndarray, moving: np.ndarray):
    ref_points, mov_points, scores = find_matches(model, reference, moving)
    minimum = 3 if TRANSFORM_TYPE == "affine" else 4
    if len(ref_points) < minimum:
        return None
    if TRANSFORM_TYPE == "affine":
        transform, inliers = cv2.estimateAffinePartial2D(
            mov_points, ref_points, method=cv2.RANSAC, ransacReprojThreshold=4.0,
            confidence=0.999, maxIters=10000)
        if transform is None:
            return None
        registered = cv2.warpAffine(moving, transform, (reference.shape[1], reference.shape[0]))
    elif TRANSFORM_TYPE == "homography":
        transform, inliers = cv2.findHomography(mov_points, ref_points, cv2.RANSAC, 4.0, confidence=0.999)
        if transform is None:
            return None
        registered = cv2.warpPerspective(moving, transform, (reference.shape[1], reference.shape[0]))
    else:
        raise ValueError("TRANSFORM_TYPE must be 'affine' or 'homography'.")
    return transform, registered, ref_points, mov_points, scores, inliers.ravel().astype(bool)


def save_results(result):
    transform, registered, ref_points, mov_points, scores, inliers = result
    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "registered_output.png"), registered)
    np.save(OUTPUT_DIR / "estimated_transform.npy", transform)
    with open(OUTPUT_DIR / "matches.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["reference_x", "reference_y", "moving_x", "moving_y", "confidence", "ransac_inlier"])
        for ref, mov, score, inlier in zip(ref_points, mov_points, scores, inliers):
            writer.writerow([*ref, *mov, float(score), int(inlier)])
    print(f"Saved registered image, transform, and matches to: {OUTPUT_DIR}")
    print(f"Candidate matches: {len(inliers)} | RANSAC inliers: {inliers.sum()} | ratio: {inliers.mean():.1%}")


def main():
    if not REFERENCE_IMAGE.exists() or not MOVING_IMAGE.exists():
        raise FileNotFoundError(
            "Add your two images in the data folder, then update REFERENCE_IMAGE and MOVING_IMAGE at the top of this file."
        )
    print(f"Using {DEVICE}. Loading pretrained LoFTR correspondence baseline.")
    print("The first run downloads pretrained weights; lunar fine-tuning needs verified correspondences.")
    model = LoFTR(pretrained="outdoor").to(DEVICE)
    reference, moving = load_grayscale(REFERENCE_IMAGE), load_grayscale(MOVING_IMAGE)
    result = register_images(model, reference, moving)
    if result is None:
        print("Registration failed: use images with overlap or lower CONFIDENCE_THRESHOLD.")
        return
    save_results(result)


if __name__ == "__main__":
    main()
