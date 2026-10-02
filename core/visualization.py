import cv2
import numpy as np

def draw_matches_visualization(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    match_result: dict,
    max_draw_matches: int = 100
) -> np.ndarray:
    """
    Renders corresponding keypoints connected with lines.
    Inlier matches are drawn in bright green, outliers in soft red.
    """
    if len(ref_img.shape) == 2:
        ref_color = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
    else:
        ref_color = ref_img.copy()

    if len(src_img.shape) == 2:
        src_color = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    else:
        src_color = src_img.copy()

    kp_ref = match_result.get("kp_ref", [])
    kp_src = match_result.get("kp_src", [])
    good_matches = match_result.get("good_matches", [])
    inliers_mask = match_result.get("inliers_mask", [])

    if not good_matches or not kp_ref or not kp_src:
        # Fallback side-by-side
        h1, w1 = ref_color.shape[:2]
        h2, w2 = src_color.shape[:2]
        target_h = max(h1, h2)
        ref_res = cv2.resize(ref_color, (w1, target_h))
        src_res = cv2.resize(src_color, (w2, target_h))
        return np.hstack([ref_res, src_res])

    # Convert inliers_mask to list of 0 and 1
    matches_mask = [int(i) for i in inliers_mask]

    # Limit matches drawn for clarity if count is large
    if len(good_matches) > max_draw_matches:
        # Prioritize inliers
        inlier_indices = [i for i, mask in enumerate(matches_mask) if mask == 1][:max_draw_matches]
        draw_matches = [good_matches[i] for i in inlier_indices]
        draw_mask = [1] * len(draw_matches)
    else:
        draw_matches = good_matches
        draw_mask = matches_mask

    canvas = cv2.drawMatches(
        src_color, kp_src,
        ref_color, kp_ref,
        draw_matches, None,
        matchColor=(0, 255, 0),       # Green for inlier matches
        singlePointColor=(255, 0, 0), # Red for un-matched keypoints
        matchesMask=draw_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )
    return canvas


def create_checkerboard(img1: np.ndarray, img2: np.ndarray, grid_size: int = 80) -> np.ndarray:
    """
    Creates an alternating checkerboard pattern of img1 (Reference) and img2 (Aligned Source).
    Helps visually detect alignment discrepancies at boundaries.
    """
    h, w = img1.shape[:2]
    
    # Standardize channels
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    checkerboard = np.zeros_like(img1)
    
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            y_end = min(y + grid_size, h)
            x_end = min(x + grid_size, w)
            if ((x // grid_size) + (y // grid_size)) % 2 == 0:
                checkerboard[y:y_end, x:x_end] = img1[y:y_end, x:x_end]
            else:
                checkerboard[y:y_end, x:x_end] = img2[y:y_end, x:x_end]

    return checkerboard


def create_color_composite(ref_img: np.ndarray, aligned_src: np.ndarray) -> np.ndarray:
    """
    Generates a False-Color Anaglyph / Composite (Red=Reference, Cyan=Aligned Source).
    Perfect alignment appears monochrome grey; misalignments appear red/cyan fringes.
    """
    if len(ref_img.shape) == 3:
        r_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    else:
        r_gray = ref_img.copy()

    if len(aligned_src.shape) == 3:
        s_gray = cv2.cvtColor(aligned_src, cv2.COLOR_RGB2GRAY)
    else:
        s_gray = aligned_src.copy()

    composite = np.zeros((r_gray.shape[0], r_gray.shape[1], 3), dtype=np.uint8)
    composite[:, :, 0] = r_gray   # Red channel = Reference
    composite[:, :, 1] = s_gray   # Green channel = Aligned Source
    composite[:, :, 2] = s_gray   # Blue channel = Aligned Source

    return composite


def create_difference_heatmap(ref_img: np.ndarray, aligned_src: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Computes absolute difference heatmap highlighting topography changes and illumination shifts.
    """
    if len(ref_img.shape) == 3:
        g1 = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    else:
        g1 = ref_img.copy()

    if len(aligned_src.shape) == 3:
        g2 = cv2.cvtColor(aligned_src, cv2.COLOR_RGB2GRAY)
    else:
        g2 = aligned_src.copy()

    diff = cv2.absdiff(g1, g2)

    if mask is not None:
        diff[mask == 0] = 0

    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    if mask is not None:
        heatmap[mask == 0] = 0

    return heatmap
