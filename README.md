# 🌕 Chandrayaan-2 Multi-Modal Lunar Image Registration Engine

**Smart India Hackathon (SIH) | Problem Statement ID: 26166**  
**Title:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2, IIRS)  
**Organization:** Indian Space Research Organisation (ISRO)

---

## 📌 Overview

Lunar surface imagery captured across different space missions (Chandrayaan-2 OHRC, TMC-2, IIRS, LRO NAC, SELENE) presents significant alignment challenges due to:
1. **Illumination & Sun Angle Differences:** Dynamic crater shadows that invert depending on solar elevation/azimuth.
2. **Scale & Resolution Variations:** Extreme spatial resolution differences (e.g. OHRC at ~0.25m/px vs TMC-2 at ~5m/px).
3. **Sensor Modalities:** High-resolution panchromatic (OHRC), stereo optical (TMC-2), and hyperspectral (IIRS).

This software tool provides an end-to-end computer vision and ML pipeline for **robust multi-modal feature detection, scale & sun-angle invariant correspondence matching, geometric image registration, and interactive overlay visualization**.

---

## 🚀 Key Features

* **Sun-Angle & Shadow Invariance:** Employs Contrast-Limited Adaptive Histogram Equalization (CLAHE), Difference of Gaussians (DoG), and Gradient Magnitude maps to neutralize deep shadow dominance.
* **Scale-Invariant Feature Extraction:** Multi-scale feature detectors (SIFT, AKAZE, ORB) with Lowe's ratio test and k-NN matcher.
* **Robust Outlier Rejection:** MAGSAC++ (USAC_MAGSAC) / RANSAC geometric estimation to eliminate false matches from repetitive crater patterns.
* **Flexible Geometric Transformations:** Homography ($3\times3$ perspective warp) and 2D Affine transformations.
* **Multi-Modal Visualizer Tools:**
  * **Alpha Blend Slider:** Dynamic opacity blending between Reference and Warped Source.
  * **Interactive Checkerboard View:** Alternating grid for inspecting border continuity.
  * **False-Color Anaglyph Composite:** Red/Cyan visual difference overlay.
  * **Difference Heatmap:** Absolute pixel intensity map showing surface structural changes.
* **Ground Control Point (GCP) Export:** Export matched coordinates as CSV or JSON for GIS mapping and cartography.

---

## 🛠️ Installation & Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd SIH-lunar-registration
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Interactive Web App:**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deploy as a Live Website

This is a **Streamlit** app, so it needs a Python server and cannot be hosted with GitHub Pages. Push this folder to GitHub, then deploy it with [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in with GitHub and choose **Create app**.
2. Select the repository and branch containing this project.
3. Set **Main file path** to `app.py` and deploy.
4. Streamlit installs `requirements.txt` and gives the app a live `https://<your-app-name>.streamlit.app` address.

The app automatically generates its demonstration image pair on the first run. Users can also upload their own lunar-image pair in the sidebar.

---

## 📂 Project Structure

```
SIH/
│── app.py                  # Main Streamlit Web Application Interface
│── generate_samples.py     # Procedural Lunar Surface Image Pair Generator
│── requirements.txt        # Python package dependencies
│── README.md               # Documentation & SIH Problem Statement details
│── core/
│   │── __init__.py
│   │── preprocessor.py     # CLAHE, DoG & Gradient feature extraction
│   │── matcher.py          # SIFT/AKAZE/ORB + MAGSAC++ Homography engine
│   │── aligner.py          # Perspective warping & GCP point extractor
│   └── visualization.py    # Checkerboard, Anaglyph, Difference Heatmap renderers
└── sample_data/            # Demo lunar image pairs (OHRC vs TMC-2)
```

---

## 🔗 Reference Data Links

* **ISRO ISSDC MAPBrowse:** [chmapbrowse.issdc.gov.in](https://chmapbrowse.issdc.gov.in/)
* **LRO NAC Downloads:** [lroc.im-ldi.com](https://lroc.im-ldi.com/images/downloads/)
* **LROC QuickMap:** [quickmap.lroc.im-ldi.com](https://quickmap.lroc.im-ldi.com/)
