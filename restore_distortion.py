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

def compute_maps(K, D, w, h, is_fisheye, undistort, alpha=1.0):
    """Helper to pre-calculate remapping maps based on camera parameters."""
    if not undistort:
        # Reverse Mode (Default): Create Distorted Image from Linear Image
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        
        if is_fisheye:
            D_fish = D[:4]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D_fish, (w, h), np.eye(3), balance=alpha)
            pts_u = cv2.fisheye.undistortPoints(pts, K, D_fish, np.eye(3), new_K)
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
            pts_u = cv2.undistortPoints(pts, K, D, None, new_K)
            
        map_coords = pts_u.reshape(h, w, 2)
        map1, map2 = cv2.convertMaps(map_coords[..., 0], map_coords[..., 1], cv2.CV_16SC2, nninterpolation=False)
    else:
        # Normal Mode (Undistort): Create Linear Image from Distorted Image
        if is_fisheye:
            D_fish = D[:4]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D_fish, (w, h), np.eye(3), balance=alpha)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D_fish, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
    return map1, map2

def get_cam_params(obj, defaults):
    """Extract camera matrices from an object with fallback to defaults."""
    w = int(obj.get('w', defaults['w']))
    h = int(obj.get('h', defaults['h']))
    fl_x = float(obj.get('fl_x', defaults['fl_x']))
    fl_y = float(obj.get('fl_y', obj.get('fl_x', defaults['fl_y'])))
    cx = float(obj.get('cx', defaults['cx']))
    cy = float(obj.get('cy', defaults['cy']))
    
    k1 = float(obj.get('k1', defaults['k1']))
    k2 = float(obj.get('k2', defaults['k2']))
    k3 = float(obj.get('k3', defaults['k3']))
    k4 = float(obj.get('k4', defaults['k4']))
    p1 = float(obj.get('p1', defaults['p1']))
    p2 = float(obj.get('p2', defaults['p2']))
    
    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, p1, p2, k3, k4, 0, 0], dtype=np.float32)
    return K, D, w, h

def main():
    args = parse_args()
    
    # 1. Load JSON Data
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        sys.exit(1)

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # 2. Extract Global Camera Parameters (Fallback)
    global_params = {
        'fl_x': float(data.get('fl_x', 1000)),
        'fl_y': float(data.get('fl_y', data.get('fl_x', 1000))),
        'cx': float(data.get('cx', 960)),
        'cy': float(data.get('cy', 540)),
        'w': int(data.get('w', 1920)),
        'h': int(data.get('h', 1080)),
        'k1': float(data.get('k1', 0)),
        'k2': float(data.get('k2', 0)),
        'k3': float(data.get('k3', 0)),
        'k4': float(data.get('k4', 0)),
        'p1': float(data.get('p1', 0)),
        'p2': float(data.get('p2', 0)),
        'is_fisheye': data.get('is_fisheye', False)
    }

    # 3. Prepare File List
    base_dir = args.image_dir if args.image_dir else os.path.dirname(os.path.abspath(args.json_path))
    files_to_process = []
    read_flags = cv2.IMREAD_COLOR if not args.exr else cv2.IMREAD_UNCHANGED

    if args.image_dir or args.exr:
        valid_exts = ('.exr',) if args.exr else ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        if os.path.exists(base_dir):
            files_to_process = [f for f in os.listdir(base_dir) if f.lower().endswith(valid_exts)]
            files_to_process.sort()
        else:
            print(f"Error: Image directory {base_dir} does not exist.")
            sys.exit(1)
    else:
        files_to_process = data.get('frames', [])

    if not files_to_process:
        print("No files to process.")
        sys.exit(0)

    # 4. Process Images with Caching
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cached_params = None
    map1, map2 = None, None

    print(f"Processing {len(files_to_process)} images...")
    for i, item in enumerate(files_to_process):
        # Determine paths and current camera params
        if isinstance(item, dict):
            # Frame from JSON
            rel_path = item['file_path'].replace('\\', os.sep).replace('/', os.sep)
            if rel_path.startswith(f'.{os.sep}'): rel_path = rel_path[2:]
            image_path = os.path.join(base_dir, rel_path)
            
            # Per-frame params or global fallback
            curr_K, curr_D, curr_w, curr_h = get_cam_params(item if 'fl_x' in item else data, global_params)
        else:
            # File from directory scan
            image_path = os.path.join(base_dir, item)
            curr_K, curr_D, curr_w, curr_h = get_cam_params(data, global_params)

        if not os.path.exists(image_path):
            continue

        # Load image (to check real resolution)
        img = cv2.imread(image_path, read_flags)
        if img is None: continue
        real_h, real_w = img.shape[:2]

        # Resolution mismatch handling
        if real_w != curr_w or real_h != curr_h:
            s_x, s_y = real_w / curr_w, real_h / curr_h
            curr_K[0,0] *= s_x; curr_K[1,1] *= s_y
            curr_K[0,2] *= s_x; curr_K[1,2] *= s_y
            curr_w, curr_h = real_w, real_h

        # Cache check & map calculation
        current_param_signature = (curr_K.tobytes(), curr_D.tobytes(), curr_w, curr_h)
        if current_param_signature != cached_params:
            map1, map2 = compute_maps(curr_K, curr_D, curr_w, curr_h, global_params['is_fisheye'], args.undistort)
            cached_params = current_param_signature
            print(f"  [Info] Parameters changed at frame {i+1}, re-calculated maps.")

        # Apply Remapping
        processed_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Save
        filename = os.path.basename(image_path)
        save_path = os.path.join(args.output_dir, filename)
        save_params = []
        if filename.lower().endswith('.exr'):
            save_params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT if processed_img.dtype == np.float32 else cv2.IMWRITE_EXR_TYPE_HALF]
        
        cv2.imwrite(save_path, processed_img, save_params)

        if (i + 1) % 50 == 0 or (i + 1) == len(files_to_process):
            print(f"  Processed {i + 1}/{len(files_to_process)} images...")

    print("Processing complete.")

    print("Processing complete.")

if __name__ == "__main__":
    main()
