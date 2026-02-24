import json
import cv2
import numpy as np
import os
from pathlib import Path
import argparse

# ==============================================================================
# 設定區域
# ==============================================================================
# 是否要裁切掉去畸變後產生的黑邊？
# True: 裁切 (視角會變窄一點點，但滿版)
# False: 保留黑邊 (視角最大化，但圖片邊緣會有黑色彎曲區域)
CROP_TO_VALID = True
# ==============================================================================

def undistort_process(json_path, output_dir, crop_to_valid):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    # 建立輸出資料夾
    output_path = Path(output_dir)
    images_out_dir = output_path / "images_undistorted"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    def get_cam_params(obj, d=None):
        """Helper to get intrinsic/distortion params from an object or fallback to defaults."""
        d = d if d else {}
        w = int(obj.get("w", d.get("w", 1920)))
        h = int(obj.get("h", d.get("h", 1080)))
        fl_x = float(obj.get("fl_x", d.get("fl_x", 1000)))
        fl_y = float(obj.get("fl_y", d.get("fl_y", fl_x)))
        cx = float(obj.get("cx", d.get("cx", w / 2)))
        cy = float(obj.get("cy", d.get("cy", h / 2)))
        k1 = float(obj.get("k1", d.get("k1", 0.0)))
        k2 = float(obj.get("k2", d.get("k2", 0.0)))
        k3 = float(obj.get("k3", d.get("k3", 0.0)))
        k4 = float(obj.get("k4", d.get("k4", 0.0)))
        p1 = float(obj.get("p1", d.get("p1", 0.0)))
        p2 = float(obj.get("p2", d.get("p2", 0.0)))
        
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array([k1, k2, p1, p2, k3, k4, 0.0, 0.0], dtype=np.float32)
        return w, h, K, D

    # Get global defaults
    global_w, global_h, global_K, global_D = get_cam_params(data)

    new_frames = []
    frames = data.get("frames", [])
    
    print(f"Processing {len(frames)} images...")

    # 準備新的 JSON 資料基礎
    new_data = data.copy()
    # 預設先把根節點的畸變參數歸零 (如果之後發現是全局共用，會在下面更新)
    for key in ["k1", "k2", "k3", "k4", "p1", "p2"]:
        if key in new_data: new_data[key] = 0.0

    json_dir = Path(json_path).parent

    for idx, frame in enumerate(frames):
        # 1. 決定這一幀使用的相機參數
        # 如果 frame 裡有 fl_x，我們就認為它有獨立參數
        if "fl_x" in frame:
            w, h, K, D = get_cam_params(frame, d=data)
            is_per_frame = True
        else:
            w, h, K, D = global_w, global_h, global_K, global_D
            is_per_frame = False

        # 2. 計算這一幀的最佳相機矩陣
        alpha = 0 if crop_to_valid else 1
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
        x, y, w_roi, h_roi = roi

        # 3. 處理路徑
        rel_path = frame["file_path"]
        img_path = json_dir / rel_path
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        # 4. 讀取並去畸變
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        dst = cv2.undistort(img, K, D, None, new_K)
        if crop_to_valid:
            dst = dst[y:y+h_roi, x:x+w_roi]

        # 5. 存檔
        img_name = Path(rel_path).name
        save_path = images_out_dir / img_name
        cv2.imwrite(str(save_path), dst)

        # 6. 更新這一幀的資料
        new_frame = frame.copy()
        new_frame["file_path"] = f"images_undistorted/{img_name}"
        
        # 更新內參
        new_frame["fl_x"] = float(new_K[0, 0])
        new_frame["fl_y"] = float(new_K[1, 1])
        new_frame["cx"] = float(new_K[0, 2])
        new_frame["cy"] = float(new_K[1, 2])
        new_frame["w"] = float(w_roi if crop_to_valid else w)
        new_frame["h"] = float(h_roi if crop_to_valid else h)
        
        # 畸變參數歸零
        for key in ["k1", "k2", "k3", "k4", "p1", "p2"]:
            if key in new_frame: new_frame[key] = 0.0
        
        # 如果是全局共用的，也順便更新根節點 (最後一幀會決定根節點數值)
        if not is_per_frame:
            new_data["fl_x"] = new_frame["fl_x"]
            new_data["fl_y"] = new_frame["fl_y"]
            new_data["cx"] = new_frame["cx"]
            new_data["cy"] = new_frame["cy"]
            new_data["w"] = new_frame["w"]
            new_data["h"] = new_frame["h"]

        new_frames.append(new_frame)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(frames):
            print(f"Processed {idx + 1}/{len(frames)}...")

    new_data["frames"] = new_frames

    # 7. 儲存新的 JSON
    new_json_path = output_path / "transforms_undistorted.json"
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    print("Done!")
    print(f"Undistorted images saved to: {images_out_dir}")
    print(f"New JSON saved to: {new_json_path}")
    print("Use this new JSON in Houdini!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort images and transforms.json")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input transforms.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--crop", dest="crop_to_valid", action="store_true", help="Crop to valid region")
    parser.add_argument("--no-crop", dest="crop_to_valid", action="store_false", help="Do not crop")
    parser.set_defaults(crop_to_valid=CROP_TO_VALID)
    
    args = parser.parse_args()
    undistort_process(args.json_path, args.output_dir, args.crop_to_valid)