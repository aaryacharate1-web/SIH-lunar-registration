import cv2
import numpy as np
import os

def create_lunar_surface_texture(size=(800, 800), seed=42):
    """
    Generates realistic procedural lunar terrain with crater impacts, sun angles, and surface regolith texture.
    """
    np.random.seed(seed)
    h, w = size
    
    # Base regolith noise using multi-octave Perlin-like noise
    base = np.zeros(size, dtype=np.float32)
    for scale in [200, 100, 50, 20, 10]:
        noise = cv2.resize(np.random.randn(h // (scale // 5), w // (scale // 5)), (w, h))
        base += noise * (scale / 200.0)

    # Normalize to [0, 255]
    base = cv2.normalize(base, None, 50, 180, cv2.NORM_MINMAX)

    # Function to stamp realistic illuminated crater
    def add_crater(img, cx, cy, radius, shadow_angle_deg=45):
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Rim and bowl profile
        crater_mask = dist <= radius * 1.3
        bowl = np.clip((1 - (dist / radius)**2), 0, 1)
        
        # Sun vector for shadow & highlight
        rad = np.radians(shadow_angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)
        gradient = (x - cx) * dx + (y - cy) * dy
        
        illumination = np.sin(np.pi * bowl) * (gradient / (radius + 1e-5)) * 120.0

        # Apply to image
        img_copy = img.copy()
        img_copy[dist <= radius] -= (bowl[dist <= radius] * 50).astype(np.float32)
        img_copy += illumination * crater_mask
        return np.clip(img_copy, 0, 255)

    # Add primary crater field
    terrain = base.astype(np.float32)
    craters = [
        (400, 400, 120), # Central major crater
        (250, 300, 60),  # Secondary crater
        (550, 220, 45),  # Secondary crater
        (600, 580, 80),  # Secondary crater
        (200, 600, 50),
        (350, 150, 35),
        (480, 500, 25),
        (150, 200, 40)
    ]

    for cx, cy, r in craters:
        terrain = add_crater(terrain, cx, cy, r, shadow_angle_deg=45)

    return terrain.astype(np.uint8)

def generate_ch2_pair_dataset(output_dir="sample_data"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Base Reference Scene (e.g. Chandrayaan-2 TMC-2 OR LRO NAC Reference)
    ref_base = create_lunar_surface_texture((800, 800), seed=101)
    
    # 2. Moving Source Scene (OHRC style: Higher resolution crop, different sun angle, rotation & scale)
    # Generate same region terrain with different sun illumination angle (135 degrees instead of 45)
    src_illum = create_lunar_surface_texture((800, 800), seed=101)
    
    # Apply geometrical transformation (Rotation, Scale, Shift) to simulate orbital perspective difference
    h, w = ref_base.shape
    center = (w // 2, h // 2)
    
    # Rotate 15 degrees, scale down 0.85x, shift (20px, -15px)
    M = cv2.getRotationMatrix2D(center, angle=12.5, scale=0.88)
    M[0, 2] += 15
    M[1, 2] -= 25
    
    src_transformed = cv2.warpAffine(src_illum, M, (w, h), flags=cv2.INTER_LANCZOS4)
    
    # Add noise & sensor blur to simulate different payload characteristics (e.g. OHRC vs TMC-2 vs IIRS)
    blur = cv2.GaussianBlur(src_transformed, (3, 3), 0.8)
    
    ref_path = os.path.join(output_dir, "ref_lunar_TMC2.png")
    src_path = os.path.join(output_dir, "src_lunar_OHRC.png")
    
    cv2.imwrite(ref_path, ref_base)
    cv2.imwrite(src_path, blur)
    print(f"Generated sample dataset: {ref_path} and {src_path}")
    return ref_path, src_path

if __name__ == "__main__":
    generate_ch2_pair_dataset()
