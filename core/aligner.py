import cv2
import numpy as np
from typing import Dict, Any, Tuple

class LunarImageAligner:
    """
    Handles geometric warping and registration of source (moving) image to reference (fixed) image space.
    """
    def __init__(self, transform_type: str = "Homography"):
        self.transform_type = transform_type

    def align_images(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        match_result: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warps src_img to align with ref_img coordinates using estimated transformation matrix.
        Returns:
            aligned_src: Aligned source image overlaid on reference footprint.
            mask: Binary mask indicating valid warped image region.
        """
        ref_h, ref_w = ref_img.shape[:2]

        if not match_result.get("success", False) or match_result.get("homography") is None:
            raise ValueError("Cannot align image: Homography matrix missing or matching failed.")

        H = match_result["homography"]

        if self.transform_type == "Affine":
            # Estimate Affine from inliers
            src_pts = match_result["src_pts"][np.array(match_result["inliers_mask"]) == 1]
            ref_pts = match_result["ref_pts"][np.array(match_result["inliers_mask"]) == 1]
            if len(src_pts) >= 3:
                A, _ = cv2.estimateAffine2D(src_pts, ref_pts)
                aligned_src = cv2.warpAffine(
                    src_img, A, (ref_w, ref_h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                mask = cv2.warpAffine(
                    np.ones(src_img.shape[:2], dtype=np.uint8) * 255, A, (ref_w, ref_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                return aligned_src, mask

        # Default Homography (Perspective)
        aligned_src = cv2.warpPerspective(
            src_img, H, (ref_w, ref_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        mask = cv2.warpPerspective(
            np.ones(src_img.shape[:2], dtype=np.uint8) * 255, H, (ref_w, ref_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return aligned_src, mask

    def extract_ground_control_points(self, match_result: Dict[str, Any]) -> list:
        """
        Extracts Ground Control Points (GCPs) from verified matching inliers.
        Returns list of dicts with source (x, y) and reference (x, y) pixel coordinates.
        """
        gcps = []
        if not match_result.get("success", False):
            return gcps

        kp_src = match_result["kp_src"]
        kp_ref = match_result["kp_ref"]
        good_matches = match_result["good_matches"]
        inliers_mask = match_result["inliers_mask"]

        for idx, is_inlier in enumerate(inliers_mask):
            if is_inlier:
                m = good_matches[idx]
                src_pt = kp_src[m.queryIdx].pt
                ref_pt = kp_ref[m.trainIdx].pt
                gcps.append({
                    "gcp_id": len(gcps) + 1,
                    "src_x": round(src_pt[0], 2),
                    "src_y": round(src_pt[1], 2),
                    "ref_x": round(ref_pt[0], 2),
                    "ref_y": round(ref_pt[1], 2),
                    "distance": round(m.distance, 4)
                })
        return gcps
