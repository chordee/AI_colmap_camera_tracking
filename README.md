# AI Colmap Camera Tracking

This project provides an automated pipeline for camera tracking and scene reconstruction using COLMAP, GLOMAP, and NeRF-compatible formats. It is designed to process video inputs, perform 3D reconstruction, and prepare the data for use in NeRF training or 3D software like Houdini.

## Features

*   **Automated Workflow:** Batch processes multiple video files.
*   **Frame Extraction:** Uses FFmpeg to extract frames from input videos.
*   **Feature Extraction & Matching:** Utilizes COLMAP for feature extraction and sequential matching.
*   **Sparse Reconstruction:** Uses **GLOMAP** for efficient sparse reconstruction.
*   **NeRF Conversion:** Converts COLMAP data to `transforms.json` (NeRF format) using `colmap2nerf.py`.
*   **Undistortion:** Includes tools to undistort images for Nerfstudio/COLMAP compatibility.
*   **Houdini Integration:** Automatically generates a Houdini (`.hip`) scene with the reconstructed point cloud and camera setups.

## Prerequisites

Ensure the following tools are installed and available in your system's PATH:

1.  **Python 3.x**
2.  **FFmpeg**: For video processing.
3.  **COLMAP**: For feature extraction and matching.
4.  **GLOMAP**: For sparse reconstruction (Mapper).
5.  **Houdini (hython)**: Required if you want to generate Houdini scenes (`build_houdini_scene.py`).

### Python Dependencies

Install the required Python packages:

```bash
pip install numpy opencv-python
```

*Optional:* For automatic object masking in `colmap2nerf.py`, you will need PyTorch and Detectron2.

## Usage

The main entry point is `run_autotracker.py`.

```bash
python run_autotracker.py <input_videos_dir> <output_dir> --scale <scale_factor> [--skip-houdini] [--hfs <houdini_path>] [--multi-cams]
```

*   `input_videos_dir`: Directory containing your source video files (e.g., `.mp4`, `.mov`).
*   `output_dir`: Directory where the results (images, sparse models, database) will be saved.
*   `--scale`: (Optional) Image scaling factor (default: `0.5`).
*   `--skip-houdini`: (Optional) Skip the generation of the Houdini `.hip` scene file.
*   `--hfs`: (Optional) Path to your Houdini installation directory (e.g., `C:\Program Files\Side Effects Software\Houdini 20.0.xxx`). If not provided, the script assumes `hython` is in your PATH.
*   `--multi-cams`: (Optional) If set, COLMAP will treat the input as multiple cameras (one per folder/video) instead of a single shared camera. Useful if videos were shot with different devices or zoom levels.

### Example

```bash
python run_autotracker.py ./videos ./output --scale 0.5 --hfs "C:/Program Files/Side Effects Software/Houdini 20.0.625"
```

## Pipeline Steps

1.  **Initialization**: The script checks/creates directories.
2.  **Tracking (`autotracker.py`)**:
    *   Extracts frames from videos.
    *   Runs COLMAP feature extraction and matching.
    *   Runs GLOMAP mapper for reconstruction.
    *   Exports the model to TXT format.
3.  **Conversion**: Converts the sparse model to PLY format.
4.  **NeRF Prep**: Runs `colmap2nerf.py` to generate `transforms.json`.
5.  **Undistortion**: Runs `undistortionNerfstudioColmap.py` to correct lens distortion.
6.  **Houdini Scene**: Runs `build_houdini_scene.py` to import the data into a Houdini file.

## Scripts Overview

*   `run_autotracker.py`: The master script that orchestrates the entire pipeline.
*   `autotracker.py`: Handles the core photogrammetry tasks (FFmpeg, COLMAP, GLOMAP).
*   `colmap2nerf.py`: Converts COLMAP data to the standard NeRF `transforms.json` format.
*   `undistortionNerfstudioColmap.py`: Handles image undistortion based on the calculated camera models.
*   `restore_distortion.py`: A utility script to restore (undistort) or reverse (distort) images based on calibration JSON. Supports EXR processing via `--exr`.
*   `build_houdini_scene.py`: Generates a `.hip` file with the point cloud and cameras loaded.

## Output Structure

For each video processed, a folder is created in the output directory containing:

*   `images/`: Extracted frames.
*   `sparse/`: COLMAP/GLOMAP sparse reconstruction data.
*   `database.db`: COLMAP database.
*   `transforms.json`: Camera poses in NeRF format.
*   `points3D.ply`: Point cloud file.
*   `undistort/`: Undistorted images and transforms.
*   `<project_name>.hip`: Houdini project file.
