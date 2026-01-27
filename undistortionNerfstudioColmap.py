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

    # 1. 讀取相機參數
    w = int(data.get("w", 1920))
    h = int(data.get("h", 1080))
    fl_x = float(data.get("fl_x", 1000))
    fl_y = float(data.get("fl_y", fl_x)) # 如果沒有 fl_y，通常預設等於 fl_x
    cx = float(data.get("cx", w / 2))
    cy = float(data.get("cy", h / 2))

    # 2. 讀取畸變參數 (Distortion Coefficients)
    k1 = float(data.get("k1", 0.0))
    k2 = float(data.get("k2", 0.0))
    k3 = float(data.get("k3", 0.0))
    k4 = float(data.get("k4", 0.0))
    p1 = float(data.get("p1", 0.0))
    p2 = float(data.get("p2", 0.0))

    # 建構相機矩陣 (Camera Matrix)
    K = np.array([
        [fl_x, 0,    cx],
        [0,    fl_y, cy],
        [0,    0,    1 ]
    ])

    # 建構畸變向量 (Distortion Vector)
    D = np.array([k1, k2, p1, p2, k3, k4, 0.0, 0.0]) # OpenCV 順序

    print(f"Camera Matrix:\n{K}")
    print(f"Distortion Coeffs: {D}")

    # 3. 計算新的相機矩陣 (Optimal New Camera Matrix)
    # 這一步很重要，因為拉直圖片後，原本的焦距和光心可能會改變
    # alpha=0: 裁切掉所有黑邊 (視角變小)
    # alpha=1: 保留所有像素 (會有黑邊)
    alpha = 0 if crop_to_valid else 1
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    
    # 用於裁切的 ROI (x, y, w, h)
    x, y, w_roi, h_roi = roi

    # 4. 準備新的 JSON 資料
    new_data = data.copy()
    
    # 更新 JSON 裡的內參為「去畸變後」的新數值
    new_data["fl_x"] = new_K[0, 0]
    new_data["fl_y"] = new_K[1, 1]
    new_data["cx"] = new_K[0, 2]
    new_data["cy"] = new_K[1, 2]
    new_data["w"] = w_roi if crop_to_valid else w
    new_data["h"] = h_roi if crop_to_valid else h
    
    # 將畸變參數歸零 (因為圖片已經直了)
    for key in ["k1", "k2", "k3", "k4", "p1", "p2"]:
        new_data[key] = 0.0

    new_frames = []
    frames = data.get("frames", [])
    
    print(f"Processing {len(frames)} images...")

    # 5. 開始批量處理圖片
    json_dir = Path(json_path).parent

    for idx, frame in enumerate(frames):
        # 處理路徑
        rel_path = frame["file_path"]
        # 嘗試組合絕對路徑
        img_path = json_dir / rel_path
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        # 讀取圖片
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 【核心步驟】去畸變
        dst = cv2.undistort(img, K, D, None, new_K)

        # 裁切 (如果 CROP_TO_VALID = True)
        if crop_to_valid:
            dst = dst[y:y+h_roi, x:x+w_roi]

        # 存檔
        img_name = Path(rel_path).name
        save_path = images_out_dir / img_name
        cv2.imwrite(str(save_path), dst)

        # 更新 frame 的 file_path 指向新圖片
        new_frame = frame.copy()
        # 這裡寫入相對路徑，方便 JSON 移動
        new_frame["file_path"] = f"images_undistorted/{img_name}"
        new_frames.append(new_frame)

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(frames)}...")

    new_data["frames"] = new_frames

    # 6. 儲存新的 JSON
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