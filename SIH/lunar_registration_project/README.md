# Lunar registration model

1. Put two overlapping lunar images in `data/`.
2. In `lunar_correspondence_model.py`, change `REFERENCE_IMAGE` and `MOVING_IMAGE` to their filenames.
3. Double-click `run_model.bat` on Windows.

Outputs appear in `outputs/`: a registered PNG, transform matrix, and a match table.

The project uses the pretrained LoFTR correspondence baseline and RANSAC registration. It can give a useful initial result, but lunar-specific fine-tuning with verified correspondences/GCPs from overlapping OHRC, TMC-2, IIRS, LRO NAC, or SELENE scenes is needed for a competition-quality model. The official SIH data endpoint is currently TBD, so no data was downloaded or assumed here.
