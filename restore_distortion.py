import json
import cv2
import numpy as np
import os
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Restore (undistort) images based on camera calibration JSON.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the .json file (e.g., transforms.json).")
    parser.add_argument("--output_dir", type=str, default="restored_output", help="Directory to save the restored images.")
    parser.add_argument("--image_dir", type=str, default=None, help="Override the directory to search for input images.")
    parser.add_argument("--undistort", action="store_true", help="Undistort images (restore). Default is to apply distortion (reverse).")
    parser.add_argument("--exr", action="store_true", help="Process .exr files in the directory instead of frames in JSON. Default off.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load JSON Data
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        sys.exit(1)

    print(f"Loading calibration data from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # 2. Extract Camera Parameters
    try:
        fl_x = data['fl_x']
        fl_y = data['fl_y']
        cx = data['cx']
        cy = data['cy']
        w = int(data['w'])
        h = int(data['h'])
        
        # Distortion coefficients
        k1 = data.get('k1', 0)
        k2 = data.get('k2', 0)
        k3 = data.get('k3', 0)
        k4 = data.get('k4', 0)
        p1 = data.get('p1', 0)
        p2 = data.get('p2', 0)
        
        is_fisheye = data.get('is_fisheye', False)
        
    except KeyError as e:
        print(f"Error: Missing critical key in JSON data: {e}")
        sys.exit(1)

    # 3. Construct Matrices
    # Camera Matrix (K)
    K = np.array([[fl_x, 0, cx],
                  [0, fl_y, cy],
                  [0, 0, 1]], dtype=np.float32)

    # Distortion Coefficients (D)
    D = np.array([k1, k2, p1, p2, k3, k4, 0, 0], dtype=np.float32)

    # Hardcoded alpha to keep all pixels
    alpha = 1.0

    print(f"  Resolution: {w}x{h}")
    print(f"  Camera Matrix (K):\n{K}")
    print(f"  Distortion Coeffs (D):\n{D}")
    print(f"  Model: {'Fisheye' if is_fisheye else 'Perspective'}")
    print(f"  Mode: {'Restore (Undistorting)' if args.undistort else 'Reverse (Distorting)'}")

    # 4. Pre-calculate Maps
    print("Pre-calculating remapping maps...")
    
    if not args.undistort:
        # Reverse Mode (Default): Create Distorted Image from Linear Image
        # We need a map: Dest(Distorted) -> Src(Linear)
        
        # 1. Create grid for Dest (Distorted)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        # Shape (N, 1, 2)
        pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        
        # 2. Map Distorted Points -> Linear Points
        if is_fisheye:
            D_fish = D[:4]
            # Estimate the linear camera matrix used in the undistorted input
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D_fish, (w, h), np.eye(3), balance=alpha
            )
            # undistortPoints: Distorted -> Linear
            pts_u = cv2.fisheye.undistortPoints(pts, K, D_fish, np.eye(3), new_K)
        else:
            # Standard Perspective
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
            # undistortPoints: Distorted -> Linear
            pts_u = cv2.undistortPoints(pts, K, D, None, new_K)
            
        map_coords = pts_u.reshape(h, w, 2)
        map1, map2 = cv2.convertMaps(map_coords[..., 0], map_coords[..., 1], cv2.CV_16SC2, nninterpolation=False)
        
    else:
        # Normal Mode (Undistort): Create Linear Image from Distorted Image
        # We need a map: Dest(Linear) -> Src(Distorted)
        
        if is_fisheye:
            D_fish = D[:4]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D_fish, (w, h), np.eye(3), balance=alpha
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D_fish, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
        else:
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D, None, new_K, (w, h), cv2.CV_16SC2
            )

    # 5. Prepare Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 6. Process Images
    # Determine base directory for images
    base_dir = args.image_dir if args.image_dir else os.path.dirname(os.path.abspath(args.json_path))

    # Prepare file list and read flags
    files_to_process = []
    read_flags = cv2.IMREAD_COLOR

    # If --image_dir is provided OR --exr is set, we scan the directory
    if args.image_dir or args.exr:
        if args.exr:
            print(f"EXR Mode Enabled: Scanning {base_dir} for .exr files...")
            read_flags = cv2.IMREAD_UNCHANGED
            valid_exts = ('.exr',)
        else:
            print(f"Scanning {base_dir} for images (ignoring JSON frames list)...")
            valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

        # Find all files with valid extensions in base_dir
        if os.path.exists(base_dir):
            files_to_process = [f for f in os.listdir(base_dir) if f.lower().endswith(valid_exts)]
            files_to_process.sort() # Ensure consistent order
        else:
            print(f"Error: Image directory {base_dir} does not exist.")
            sys.exit(1)
            
        if not files_to_process:
             print(f"No matching image files found in {base_dir}")
             sys.exit(0)
             
    else:
        # Fallback to JSON frames if no image_dir override
        frames = data.get('frames', [])
        if not frames:
            print("No frames found in JSON 'frames' list.")
            sys.exit(0)
        files_to_process = frames

    print(f"Starting processing of {len(files_to_process)} images...")

    for i, item in enumerate(files_to_process):
        # Determine image path based on mode
        if args.image_dir or args.exr:
            # item is just the filename
            rel_path = item
            image_path = os.path.join(base_dir, rel_path)
        else:
            # item is a frame dict from JSON
            rel_path = item['file_path']
            rel_path = rel_path.replace('\\', os.sep).replace('/', os.sep)
            
            # If path starts with ./, remove it to join cleanly
            if rel_path.startswith(f'.{os.sep}'):
                rel_path = rel_path[2:]
                
            image_path = os.path.join(base_dir, rel_path)
        
        if not os.path.exists(image_path):
            print(f"  [Skipping] Image not found: {image_path}")
            continue

        # Read image
        img = cv2.imread(image_path, read_flags)
        if img is None:
            print(f"  [Skipping] Could not read image: {image_path}")
            continue

        # Perform Remapping (works for both directions now)
        processed_img = cv2.remap(
            img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # Save result
        filename = os.path.basename(image_path)
        save_path = os.path.join(args.output_dir, filename)
        
        save_params = []
        if filename.lower().endswith('.exr'):
            # Attempt to match the output EXR type to the processing data type
            if processed_img.dtype == np.float32:
                save_params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
            elif processed_img.dtype == np.float16:
                save_params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]

        cv2.imwrite(save_path, processed_img, save_params)
        
        # Simple progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(files_to_process)} images...")

    print("Processing complete.")

if __name__ == "__main__":
    main()
