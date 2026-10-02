import cv2
import numpy as np
import math
from typing import Tuple, List, Dict, Any

class MultiModalLunarMatcher:
    """
    Multi-modal, Sun angle, and scale invariant correspondence matcher
    designed for Chandrayaan-2 (OHRC, TMC-2, IIRS) and reference images (LRO NAC, SELENE).
    """
    def __init__(self, method: str = "SIFT", ratio_thresh: float = 0.75, ransac_reproj_thresh: float = 4.0):
        self.method = method.upper()
        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self._init_detector()

    def _init_detector(self):
        if self.method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=10)
            self.norm_type = cv2.NORM_L2
        elif self.method == "AKAZE":
            self.detector = cv2.AKAZE_create(threshold=0.0005)
            self.norm_type = cv2.NORM_HAMMING
        elif self.method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
            self.norm_type = cv2.NORM_HAMMING
        else:
            # Default to SIFT
            self.detector = cv2.SIFT_create(nfeatures=5000)
            self.norm_type = cv2.NORM_L2

    def match_features(
        self,
        ref_img: np.ndarray,
        src_img: np.ndarray,
        ref_prep: np.ndarray = None,
        src_prep: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Detects keypoints and matches features between reference (fixed) and source (moving) lunar images.
        """
        # Select input for feature detection: preprocessed feature maps work best for cross-illumination
        ref_input = ref_prep if ref_prep is not None else ref_img
        src_input = src_prep if src_prep is not None else src_img

        # Detect keypoints and compute descriptors
        kp_ref, des_ref = self.detector.detectAndCompute(ref_input, None)
        kp_src, des_src = self.detector.detectAndCompute(src_input, None)

        if des_ref is None or des_src is None or len(kp_ref) < 4 or len(kp_src) < 4:
            return {
                "success": False,
                "error": "Insufficient keypoints detected in one or both images.",
                "num_ref_kp": len(kp_ref) if kp_ref else 0,
                "num_src_kp": len(kp_src) if kp_src else 0,
                "good_matches": [],
                "inliers_count": 0,
                "homography": None
            }

        # Matcher instantiation
        if self.norm_type == cv2.NORM_L2:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # k-NN Matching with Lowe's Ratio Test
        knn_matches = matcher.knnMatch(des_src, des_ref, k=2)
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return {
                "success": False,
                "error": f"Only {len(good_matches)} match candidates found after ratio test (minimum 4 required).",
                "num_ref_kp": len(kp_ref),
                "num_src_kp": len(kp_src),
                "good_matches": good_matches,
                "inliers_count": 0,
                "homography": None
            }

        # Extract coordinates of matches
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Geometric outlier rejection using MAGSAC++ or RANSAC
        ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, mask = cv2.findHomography(src_pts, ref_pts, ransac_method, self.ransac_reproj_thresh)

        inliers_mask = mask.ravel().tolist() if mask is not None else []
        inliers_count = int(np.sum(mask)) if mask is not None else 0

        # Calculate Mean Reprojection Error for inliers
        reproj_error = 0.0
        if H is not None and inliers_count > 0:
            inlier_src = src_pts[mask.ravel() == 1]
            inlier_ref = ref_pts[mask.ravel() == 1]
            projected_src = cv2.perspectiveTransform(inlier_src, H)
            errors = np.linalg.norm(projected_src - inlier_ref, axis=2)
            reproj_error = float(np.mean(errors))

        return {
            "success": H is not None and inliers_count >= 4,
            "num_ref_kp": len(kp_ref),
            "num_src_kp": len(kp_src),
            "kp_ref": kp_ref,
            "kp_src": kp_src,
            "good_matches": good_matches,
            "inliers_mask": inliers_mask,
            "inliers_count": inliers_count,
            "match_ratio": round((inliers_count / len(good_matches)) * 100, 2) if good_matches else 0,
            "homography": H,
            "reproj_error_px": round(reproj_error, 3),
            "src_pts": src_pts,
            "ref_pts": ref_pts
        }
