import cv2
import numpy as np

class LunarImagePreprocessor:
    """
    Preprocessor for Chandrayaan-2 and reference lunar optical images.
    Addresses Sun angle variations, shadow dominance, and scale differences.
    """
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE and shadow-mitigation filters to equalize extreme lunar shadow contrast.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Standardize range [0, 255] uint8
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # CLAHE contrast enhancement
        enhanced = self.clahe.apply(gray)
        return enhanced

    def extract_sun_invariant_features_map(self, image: np.ndarray) -> np.ndarray:
        """
        Generates a phase/edge invariant representation suitable for multi-modal matching
        where shadow directions differ due to different sun angles.
        """
        enhanced = self.normalize_illumination(image)
        
        # Difference of Gaussians (DoG) for scale-space illumination independence
        g1 = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
        g2 = cv2.GaussianBlur(enhanced, (9, 9), 3.0)
        dog = cv2.subtract(g1, g2)
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Gradient Magnitude (Sobel) to capture structural crater rims & ridges
        sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Combine DoG and Gradient magnitude
        combined = cv2.addWeighted(dog, 0.5, magnitude, 0.5, 0)
        return combined

    def resize_to_common_scale(self, img_ref: np.ndarray, img_src: np.ndarray, max_dim: int = 1200):
        """
        Resizes images to manageable processing dimensions while preserving aspect ratios.
        """
        def scale_img(img):
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                new_w, new_h = int(w * scale), int(h * scale)
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
            return img, 1.0

        ref_scaled, scale_ref = scale_img(img_ref)
        src_scaled, scale_src = scale_img(img_src)
        return ref_scaled, src_scaled, scale_ref, scale_src
