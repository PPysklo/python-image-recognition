# Image Recognition with SIFT and DBSCAN

This Python script performs image recognition using SIFT (Scale-Invariant Feature Transform) and DBSCAN (Density-Based Spatial Clustering). It detects and locates a given pattern image within a target image or video by identifying key points and clustering them.

## Features

Feature Detection with SIFT: Extracts key points and descriptors.

FLANN-based Matching: Matches features between images.

Lowe's Ratio Test: Filters incorrect matches.

DBSCAN Clustering: Identifies the most significant feature region.

Bounding Box Highlighting: Draws a rectangle around the detected pattern.

Supports Image and Video Processing: Works with static images and .MOV video files.

Graphical File Selection: Uses Tkinter for an easy-to-use file picker.

## Requirements

Ensure you have the following dependencies installed:
```bash
pip install numpy opencv-python scikit-learn pillow
```
## Usage

Run the script and select a pattern image and a target image or video when prompted.

```bash
python image_recognition.py
```
## Controls

If using a video file, press 'k' to stop processing.

The script will display the detected area with a bounding box if a match is found.

## Example Output

When a pattern is detected, the program highlights the region:
