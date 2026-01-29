import os
import sys
import subprocess
import glob
import argparse

# System Binaries (Ensure these are in your PATH)
FFMPEG = "ffmpeg"
COLMAP = "colmap"
GLOMAP = "glomap"

def run_command(cmd, error_msg, quiet=False):
    """Runs a subprocess command. Returns True on success, False on failure."""
    try:
        kwargs = {}
        if quiet:
            kwargs['stdout'] = subprocess.DEVNULL
            kwargs['stderr'] = subprocess.DEVNULL
        
        # Run command
        # print(f"DEBUG: Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, **kwargs)
        return True
    except subprocess.CalledProcessError:
        print(error_msg)
        return False
    except FileNotFoundError:
        print(f"        [ERROR] Binary not found: {cmd[0]}")
        print(error_msg)
        return False

def process_video(video_path, scenes_dir, idx, total, overlap=12, scale=1.0, mask_path=None, multi_cams=False, acescg=False, lut_path=None):
    # Get base name and extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ext = os.path.splitext(video_path)[1]

    print(f"\n[{idx}/{total}] === Processing \"{base_name}{ext}\" ===")

    # Directory layout
    scene_path = os.path.join(scenes_dir, base_name)
    img_dir = os.path.join(scene_path, "images")
    sparse_dir = os.path.join(scene_path, "sparse")
    database_path = os.path.join(scene_path, "database.db")

    # Skip if already reconstructed
    if os.path.exists(scene_path):
        print(f"        • Skipping \"{base_name}\" – already reconstructed.")
        return

    # Clean slate
    try:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
    except OSError as e:
        print(f"        [ERROR] Could not create directories: {e}")
        return

    # 1) Extract every frame
    print("        [1/4] Extracting frames ...")
    frame_pattern = os.path.join(img_dir, "frame_%06d.jpg")
    cmd_ffmpeg = [
        FFMPEG, "-loglevel", "error", "-stats", "-i", video_path,
        "-qscale:v", "2"
    ]
    
    # Build video filters
    filters = []
    
    # ACEScg to sRGB conversion (Generic transform using zscale)
    if acescg:
        # tin=linear (Linear input), t=iec61966-2-1 (sRGB EOTF output)
        # pin=bt2020 (ACEScg is AP1, bt2020 is closest standard primary in zscale)
        # p=bt709 (sRGB/Rec709 primaries)
        filters.append("zscale=tin=linear:t=iec61966-2-1:pin=bt2020:p=bt709:min=bt2020nc:m=bt709")

    # Apply LUT if provided
    if lut_path:
        # Use lut3d filter for .cube files
        safe_lut_path = lut_path.replace("\\", "/") # FFmpeg filters prefer forward slashes
        filters.append(f"lut3d='{safe_lut_path}'")

    if scale != 1.0:
        filters.append(f"scale=iw*{scale}:ih*{scale}")

    if filters:
        cmd_ffmpeg.extend(["-vf", ",".join(filters)])

    cmd_ffmpeg.append(frame_pattern)

    if not run_command(cmd_ffmpeg, f"        × FFmpeg failed – skipping \"{base_name}\"."):
        return

    # Check if frames were extracted
    if not glob.glob(os.path.join(img_dir, "*.jpg")):
        print(f"        × No frames extracted – skipping \"{base_name}\".")
        return

    # 2) Feature extraction (COLMAP)
    print("        [2/4] COLMAP feature_extractor ...")
    cmd_colmap_fe = [
        COLMAP, "feature_extractor",
        "--database_path", database_path,
        "--image_path", img_dir,
        "--SiftExtraction.use_gpu", "1"
    ]
    
    if multi_cams:
        cmd_colmap_fe.extend(["--ImageReader.single_camera_per_folder", "1"])
    else:
        cmd_colmap_fe.extend(["--ImageReader.single_camera", "1"])

    if mask_path:
        cmd_colmap_fe.extend(["--ImageReader.mask_path", mask_path])

    if not run_command(cmd_colmap_fe, f"        × feature_extractor failed – skipping \"{base_name}\"."):
        return

    # 3) Sequential matching (COLMAP)
    print("        [3/4] COLMAP sequential_matcher ...")
    cmd_colmap_sm = [
        COLMAP, "sequential_matcher",
        "--database_path", database_path,
        "--SequentialMatching.overlap", str(overlap)
    ]
    if not run_command(cmd_colmap_sm, f"        × sequential_matcher failed – skipping \"{base_name}\"."):
        return

    # 4) Sparse reconstruction (GLOMAP)
    print("        [4/4] GLOMAP mapper ...")
    cmd_glomap = [
        GLOMAP, "mapper",
        "--database_path", database_path,
        "--image_path", img_dir,
        "--output_path", sparse_dir
    ]
    if not run_command(cmd_glomap, f"        × glomap mapper failed – skipping \"{base_name}\"."):
        return

    # Export TXT inside the model folder
    # Keep TXT next to BIN so Blender can import from sparse\0 directly.
    sparse_0_dir = os.path.join(sparse_dir, "0")
    if os.path.exists(sparse_0_dir):
        cmd_convert_1 = [
            COLMAP, "model_converter",
            "--input_path", sparse_0_dir,
            "--output_path", sparse_0_dir,
            "--output_type", "TXT"
        ]
        run_command(cmd_convert_1, "        [WARN] Failed to export TXT to sparse/0", quiet=True)

        # Export TXT to parent sparse\ (for Blender auto-detect)
        cmd_convert_2 = [
            COLMAP, "model_converter",
            "--input_path", sparse_0_dir,
            "--output_path", sparse_dir,
            "--output_type", "TXT"
        ]
        run_command(cmd_convert_2, "        [WARN] Failed to export TXT to sparse/", quiet=True)

    print(f"        ✓ Finished \"{base_name}\"  ({idx}/{total})")

def main():
    parser = argparse.ArgumentParser(description="Batch script for automated photogrammetry tracking workflow.")
    parser.add_argument("videos_dir", help="Directory containing input videos")
    parser.add_argument("scenes_dir", help="Directory to output scenes")
    parser.add_argument("--overlap", type=int, default=12, help="Sequential matching overlap (default: 12)")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scaling factor (default: 1.0)")
    parser.add_argument("--mask", help="Path to mask directory (optional)")
    parser.add_argument("--multi-cams", action="store_true", help="Allow processing multiple videos with different camera settings")
    parser.add_argument("--acescg", action="store_true", help="Convert input ACEScg colorspace to sRGB")
    parser.add_argument("--lut", help="Path to .cube LUT file for color conversion (optional)")
    
    # If no arguments provided, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    
    videos_dir = os.path.abspath(args.videos_dir)
    scenes_dir = os.path.abspath(args.scenes_dir)
    mask_path = os.path.abspath(args.mask) if args.mask else None
    lut_path = os.path.abspath(args.lut) if args.lut else None

    # Ensure required folders exist
    if not os.path.isdir(videos_dir):
        print(f"[ERROR] Input folder \"{videos_dir}\" missing.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    try:
        os.makedirs(scenes_dir, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Could not create output folder \"{scenes_dir}\": {e}")
        input("Press Enter to exit...")
        sys.exit(1)

    # Count videos
    # Filter for files only
    video_files = [f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))]
    total = len(video_files)

    if total == 0:
        print(f"[INFO] No video files found in \"{videos_dir}\".")
        input("Press Enter to exit...")
        sys.exit(0)

    print("==============================================================")
    print(f" Starting GLOMAP pipeline on {total} video(s) ...")
    print("==============================================================")

    for idx, video_file in enumerate(video_files, 1):
        process_video(
            os.path.join(videos_dir, video_file), 
            scenes_dir, 
            idx, 
            total, 
            overlap=args.overlap, 
            scale=args.scale, 
            mask_path=mask_path, 
            multi_cams=args.multi_cams,
            acescg=args.acescg,
            lut_path=lut_path
        )

    print("--------------------------------------------------------------")
    print(f" All jobs finished – results are in \"{scenes_dir}\".")
    print("--------------------------------------------------------------")

if __name__ == "__main__":
    main()
