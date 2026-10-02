import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import json
import os

from core.preprocessor import LunarImagePreprocessor
from core.matcher import MultiModalLunarMatcher
from core.aligner import LunarImageAligner
from core.visualization import (
    draw_matches_visualization,
    create_checkerboard,
    create_color_composite,
    create_difference_heatmap
)
from generate_samples import generate_ch2_pair_dataset

# Page Configuration
st.set_page_config(
    page_title="ISRO Chandrayaan-2 Lunar Image Registration Tool",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for Space/ISRO theme
st.markdown("""
<style>
    .main {
        background-color: #0b0e14;
        color: #e0e6ed;
    }
    .stApp {
        background-color: #0b0e14;
    }
    .css-1d3 Sterling {
        background-color: #121824;
    }
    .title-header {
        text-align: center;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #38bdf8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #0284c7, #2563eb);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0369a1, #1d4ed8);
    }
</style>
""", unsafe_allow_html=True)

# Ensure sample data exists
SAMPLE_DIR = "sample_data"
ref_sample_path = os.path.join(SAMPLE_DIR, "ref_lunar_TMC2.png")
src_sample_path = os.path.join(SAMPLE_DIR, "src_lunar_OHRC.png")
if not os.path.exists(ref_sample_path) or not os.path.exists(src_sample_path):
    generate_ch2_pair_dataset(SAMPLE_DIR)

# App Header
st.markdown("""
<div class="title-header">
    <h1 style="color: #f8fafc; margin-bottom: 5px;">🌕 Multi-Modal Lunar Image Registration Engine</h1>
    <h3 style="color: #38bdf8; font-weight: 400; margin-top: 0;">Chandrayaan-2 OHRC, TMC-2 & Reference Payload Correspondence Tool</h3>
    <p style="color: #94a3b8; font-size: 0.95rem;">
        Smart India Hackathon (SIH Problem Statement ID: 26166) | Sun Angle & Scale Invariant Surface Feature Correspondence
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Matching Parameters")

data_source = st.sidebar.radio(
    "Dataset Selection",
    ["Use Chandrayaan-2 Sample Pair (OHRC vs TMC-2)", "Upload Custom Lunar Image Pair"]
)

st.sidebar.subheader("Preprocessing Options")
enable_clahe = st.sidebar.checkbox("Sun Angle / Illumination CLAHE Equalization", value=True)
enable_dog = st.sidebar.checkbox("DoG Invariant Ridge/Edge Feature Map", value=True)

st.sidebar.subheader("Matching Algorithm")
match_method = st.sidebar.selectbox("Feature Detector", ["SIFT", "AKAZE", "ORB"], index=0)
ratio_threshold = st.sidebar.slider("Lowe's Ratio Test Threshold", 0.50, 0.90, 0.75, step=0.05)
ransac_thresh = st.sidebar.slider("MAGSAC++ Reprojection Error (px)", 1.0, 10.0, 4.0, step=0.5)
transform_type = st.sidebar.selectbox("Geometric Transformation", ["Homography", "Affine"], index=0)

# Helper function to load image
def load_image(source) -> np.ndarray:
    if isinstance(source, str):
        img = cv2.imread(source)
    else:
        file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        source.seek(0)
    
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Load Images
ref_img = None
src_img = None

if data_source == "Use Chandrayaan-2 Sample Pair (OHRC vs TMC-2)":
    ref_img = load_image(ref_sample_path)
    src_img = load_image(src_sample_path)
    ref_name = "Chandrayaan-2 TMC-2 (Fixed Reference)"
    src_name = "Chandrayaan-2 OHRC (Moving Source)"
else:
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        u_ref = st.file_uploader("Upload Reference Image (Fixed)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if u_ref:
            ref_img = load_image(u_ref)
            ref_name = u_ref.name
    with col_u2:
        u_src = st.file_uploader("Upload Source Image (Moving)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if u_src:
            src_img = load_image(u_src)
            src_name = u_src.name

if ref_img is None or src_img is None:
    st.info("👆 Please upload Reference and Source lunar images in the sidebar or select the pre-loaded Chandrayaan-2 sample pair.")
    st.stop()

# Initialize Engine
preprocessor = LunarImagePreprocessor()
matcher = MultiModalLunarMatcher(method=match_method, ratio_thresh=ratio_threshold, ransac_reproj_thresh=ransac_thresh)
aligner = LunarImageAligner(transform_type=transform_type)

# Preprocessing Step
ref_prep = None
src_prep = None

if enable_dog:
    ref_prep = preprocessor.extract_sun_invariant_features_map(ref_img)
    src_prep = preprocessor.extract_sun_invariant_features_map(src_img)
elif enable_clahe:
    ref_prep = preprocessor.normalize_illumination(ref_img)
    src_prep = preprocessor.normalize_illumination(src_img)

# Perform Matching
match_res = matcher.match_features(ref_img, src_img, ref_prep=ref_prep, src_prep=src_prep)

# Perform Alignment if matched successfully
aligned_src = None
alignment_mask = None
gcps = []

if match_res["success"]:
    aligned_src, alignment_mask = aligner.align_images(src_img, ref_img, match_res)
    gcps = aligner.extract_ground_control_points(match_res)

# Navigation Tabs
tab_overview, tab_matches, tab_align, tab_compare, tab_gcps = st.tabs([
    "📍 1. Image Pair Overview",
    "🔍 2. Sun-Invariant Matches",
    "📐 3. Transformation & Homography",
    "👁️ 4. Multi-Modal Visualizer",
    "💾 5. Ground Control Points & Export"
])

# TAB 1: OVERVIEW
with tab_overview:
    st.subheader("Input Payload Pair")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Reference Image (Fixed Target)**: `{ref_name}`")
        st.image(ref_img, use_container_width=True, caption=f"Resolution: {ref_img.shape[1]}x{ref_img.shape[0]} px")
    with c2:
        st.markdown(f"**Source Image (Moving Payload)**: `{src_name}`")
        st.image(src_img, use_container_width=True, caption=f"Resolution: {src_img.shape[1]}x{src_img.shape[0]} px")

    st.markdown("---")
    st.subheader("Preprocessed Sun-Invariant Feature Maps")
    st.markdown("Normalizes high contrast lunar crater shadows caused by varying solar azimuth angles.")
    cp1, cp2 = st.columns(2)
    with cp1:
        if ref_prep is not None:
            st.image(ref_prep, use_container_width=True, caption="Reference Invariant Feature Map (DoG / Sobel)")
        else:
            st.image(ref_img, use_container_width=True, caption="Raw Reference Image")
    with cp2:
        if src_prep is not None:
            st.image(src_prep, use_container_width=True, caption="Source Invariant Feature Map (DoG / Sobel)")
        else:
            st.image(src_img, use_container_width=True, caption="Raw Source Image")

# TAB 2: MATCHES
with tab_matches:
    st.subheader("Feature Detection & Correspondence")
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Reference Keypoints", match_res["num_ref_kp"])
    with m2:
        st.metric("Source Keypoints", match_res["num_src_kp"])
    with m3:
        st.metric("MAGSAC++ Inliers", match_res["inliers_count"])
    with m4:
        st.metric("Reprojection Error", f"{match_res['reproj_error_px']} px")

    if not match_res["success"]:
        st.error(f"❌ Matching failed: {match_res.get('error', 'Insufficient geometric inliers.')}")
    else:
        st.success(f"✅ Inlier Ratio: **{match_res['match_ratio']}%** ({match_res['inliers_count']} valid correspondences found).")
        
        matches_vis = draw_matches_visualization(ref_img, src_img, match_res)
        st.image(matches_vis, use_container_width=True, caption="Green lines represent verified geometrically consistent surface matches.")

# TAB 3: HOMOGRAPHY & TRANSFORM
with tab_align:
    st.subheader("Geometric Registration Parameters")
    if match_res["success"] and match_res["homography"] is not None:
        H = match_res["homography"]
        st.markdown(f"### Estimated Matrix ({transform_type})")
        df_H = pd.DataFrame(H, columns=["h1", "h2", "h3"])
        st.dataframe(df_H.style.format("{:.6f}"))

        st.markdown("### Aligned Footprint Output")
        ca1, ca2 = st.columns(2)
        with ca1:
            st.image(ref_img, use_container_width=True, caption="Target Reference Terrain (Fixed)")
        with ca2:
            st.image(aligned_src, use_container_width=True, caption=f"Warped Source Image ({transform_type} Aligned)")
    else:
        st.warning("No transformation matrix available. Adjust matching parameters in sidebar.")

# TAB 4: MULTI-MODAL VISUALIZER
with tab_compare:
    st.subheader("Interactive Alignment Inspection Tools")
    
    if aligned_src is not None:
        mode = st.radio(
            "Select Visualization Inspection Mode",
            ["Alpha Blend Overlay", "Checkerboard Grid", "False-Color Anaglyph Composite", "Topological Difference Heatmap"],
            horizontal=True
        )

        if mode == "Alpha Blend Overlay":
            alpha = st.slider("Source / Reference Blend Alpha", 0.0, 1.0, 0.5, step=0.05)
            
            # Match channel dimensions
            r_c = ref_img if len(ref_img.shape) == 3 else cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
            s_c = aligned_src if len(aligned_src.shape) == 3 else cv2.cvtColor(aligned_src, cv2.COLOR_GRAY2RGB)
            
            blended = cv2.addWeighted(r_c, 1 - alpha, s_c, alpha, 0)
            st.image(blended, use_container_width=True, caption=f"Alpha Blend ({int((1-alpha)*100)}% Reference, {int(alpha*100)}% Aligned Source)")

        elif mode == "Checkerboard Grid":
            grid_size = st.slider("Checkerboard Grid Square Size (px)", 20, 200, 80, step=10)
            checkerboard = create_checkerboard(ref_img, aligned_src, grid_size=grid_size)
            st.image(checkerboard, use_container_width=True, caption="Grid alignment check: seamless terrain features indicate perfect registration.")

        elif mode == "False-Color Anaglyph Composite":
            composite = create_color_composite(ref_img, aligned_src)
            st.image(composite, use_container_width=True, caption="False Color: Red = Reference, Cyan = Aligned Source. Alignment = Monochrome grey.")

        elif mode == "Topological Difference Heatmap":
            heatmap = create_difference_heatmap(ref_img, aligned_src, mask=alignment_mask)
            st.image(heatmap, use_container_width=True, caption="Absolute pixel intensity difference heatmap highlighting shadow shifts & topographic changes.")
    else:
        st.info("Registration results required to view multi-modal visualizer.")

# TAB 5: GCP EXPORT & DOWNLOAD
with tab_gcps:
    st.subheader("Ground Control Points (GCPs) & Export Data")
    if gcps:
        df_gcps = pd.DataFrame(gcps)
        st.dataframe(df_gcps, use_container_width=True)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            csv_data = df_gcps.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download GCPs (CSV)", data=csv_data, file_name="lunar_gcps.csv", mime="text/csv")
        with col_d2:
            json_data = json.dumps(gcps, indent=2)
            st.download_button("📥 Download GCPs (JSON)", data=json_data, file_name="lunar_gcps.json", mime="application/json")
        with col_d3:
            if aligned_src is not None:
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(aligned_src, cv2.COLOR_RGB2BGR))
                if is_success:
                    st.download_button("📥 Download Aligned Image (PNG)", data=buffer.tobytes(), file_name="aligned_lunar_source.png", mime="image/png")
    else:
        st.info("No Ground Control Points available for download.")
