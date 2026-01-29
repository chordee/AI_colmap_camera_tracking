import argparse
import os
import subprocess
import sys
import shutil

def main():
    parser = argparse.ArgumentParser(description="Batch runner for autotracker and colmap conversion.")
    parser.add_argument("input_path", help="Path to input directory (videos)")
    parser.add_argument("output_path", help="Path to output directory")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale argument (default: 0.5)")
    parser.add_argument("--skip-houdini", action="store_true", help="Skip Houdini scene generation")
    parser.add_argument("--hfs", help="Path to Houdini installation (optional)")
    parser.add_argument("--multi-cams", action="store_true", help="Allow processing multiple videos with different camera settings")
    parser.add_argument("--acescg", action="store_true", help="Convert input ACEScg colorspace to sRGB")
    parser.add_argument("--lut", help="Path to .cube LUT file for color conversion (optional)")
    parser.add_argument("--mask", help="Path to mask directory root (optional)")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    scale = args.scale

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path, exist_ok=True)
            print(f"[INFO] Created output directory: {output_path}")
        except OSError as e:
            print(f"[ERROR] Could not create output directory: {e}")
            sys.exit(1)

    # Locate autotracker.py (assumed to be in the same directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autotracker_script = os.path.join(script_dir, "autotracker.py")

    # Command 1: python autotracker.py <input_path> <output_path> --scale <scale>
    cmd1 = [sys.executable, autotracker_script, input_path, output_path, "--scale", str(scale)]
    if args.multi_cams:
        cmd1.append("--multi-cams")
    if args.acescg:
        cmd1.append("--acescg")
    if args.lut:
        cmd1.extend(["--lut", args.lut])
    if args.mask:
        cmd1.extend(["--mask", args.mask])
        
    print(f"Running: {' '.join(cmd1)}")
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] autotracker.py failed.")
        sys.exit(1)

    # Command 2: Run colmap model_converter on subfolders
    print("Scanning output directory for subfolders to convert models...")
    subfolders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
    
    for folder in subfolders:
        folder_path = os.path.join(output_path, folder)
        sparse_0_path = os.path.join(folder_path, "sparse", "0")
        ply_output_path = os.path.join(folder_path, "points3D.ply")
        
        if os.path.exists(sparse_0_path):
            cmd2 = ["colmap", "model_converter", "--input_path", sparse_0_path, "--output_path", ply_output_path, "--output_type", "PLY"]
            print(f"Running: {' '.join(cmd2)}")
            try:
                subprocess.run(cmd2, check=True)
            except subprocess.CalledProcessError:
                print(f"[ERROR] colmap model_converter failed for {folder}.")

    # Command 3: Copy colmap2nerf.py to output_path
    colmap2nerf_src = os.path.join(script_dir, "colmap2nerf.py")
    colmap2nerf_dst = os.path.join(output_path, "colmap2nerf.py")
    print(f"Copying {colmap2nerf_src} to {colmap2nerf_dst}")
    try:
        shutil.copy(colmap2nerf_src, colmap2nerf_dst)
    except OSError as e:
        print(f"[ERROR] Failed to copy colmap2nerf.py: {e}")
        sys.exit(1)

    # Command 4: Switch workspace and run colmap2nerf on subfolders
    original_cwd = os.getcwd()
    os.chdir(output_path)
    print(f"Switched workspace to: {os.getcwd()}")

    generated_jsons = []
    subfolders = [f for f in os.listdir(".") if os.path.isdir(f)]
    for folder in subfolders:
        print(f"Processing folder: {folder}")
        json_filename = f"{folder}_transforms.json"
        cmd_nerf = [
            sys.executable, "colmap2nerf.py",
            "--colmap_db", os.path.join(folder, "database.db"),
            "--images", os.path.join(folder, "images"),
            "--text", os.path.join(folder, "sparse"),
            "--out", json_filename,
            "--keep_colmap_coords"
        ]
        print(f"Running: {' '.join(cmd_nerf)}")
        subprocess.run(cmd_nerf, check=False)
        
        if os.path.exists(json_filename):
            generated_jsons.append((os.path.abspath(json_filename), folder))

    # Command 5: Back to original workspace and run undistortion
    os.chdir(original_cwd)
    print(f"Switched workspace back to: {os.getcwd()}")

    undistortion_script = os.path.join(script_dir, "undistortionNerfstudioColmap.py")

    for json_path, folder_name in generated_jsons:
        undistort_output_dir = os.path.join(output_path, folder_name, "undistort")
        cmd_undistort = [
            sys.executable, undistortion_script,
            "--no-crop",
            "--json_path", json_path,
            "--output_dir", undistort_output_dir
        ]
        print(f"Running: {' '.join(cmd_undistort)}")
        try:
            subprocess.run(cmd_undistort, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] undistortionNerfstudioColmap.py failed for {folder_name}.")

    # Command 6: Run build_houdini_scene.py
    if not args.skip_houdini:
        houdini_script = os.path.join(script_dir, "build_houdini_scene.py")
        print("Scanning output directory for Houdini scene generation...")

        # Determine hython executable
        if args.hfs:
            hython_exec = os.path.join(args.hfs, "bin", "hython")
            if sys.platform == "win32":
                hython_exec += ".exe"
        else:
            hython_exec = "hython"
        
        subfolders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
        for folder in subfolders:
            folder_path = os.path.join(output_path, folder)
            ply_path = os.path.join(folder_path, "points3D.ply").replace("\\", "/")
            undistort_dir = os.path.join(folder_path, "undistort")
            json_path = os.path.join(undistort_dir, "transforms_undistorted.json").replace("\\", "/")
            hip_path = os.path.join(folder_path, f"{folder}.hip").replace("\\", "/")

            if os.path.exists(ply_path) and os.path.exists(json_path):
                cmd_houdini = [hython_exec, houdini_script, json_path, ply_path, hip_path]
                print(f"Running: {' '.join(cmd_houdini)}")
                try:
                    subprocess.run(cmd_houdini, check=True)
                except subprocess.CalledProcessError:
                    print(f"[ERROR] build_houdini_scene.py failed for {folder}.")
                except FileNotFoundError:
                    print(f"[ERROR] {hython_exec} executable not found. Ensure Houdini is in PATH or --hfs is correct.")
    else:
        print("Skipping Houdini scene generation as per argument.")

if __name__ == "__main__":
    main()