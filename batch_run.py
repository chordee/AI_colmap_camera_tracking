import os
import subprocess
import sys
import argparse
import configparser
import json

def main():
    parser = argparse.ArgumentParser(description="Batch run autotracker on subdirectories of a target path.")
    parser.add_argument("target_path", nargs="?", default=".", help="The path to scan for directories (default: current directory)")
    
    # Sync arguments with run_autotracker.py
    parser.add_argument("--scale", type=float, help="Default scale argument")
    parser.add_argument("--overlap", type=int, help="Default sequential matching overlap")
    parser.add_argument("--camera_model", help="Default COLMAP camera model")
    parser.add_argument("--mask", help="Default mask directory root")
    parser.add_argument("--lut", help="Default path to .cube LUT file")
    parser.add_argument("--hfs", help="Default path to Houdini installation")
    parser.add_argument("--multi-cams", action="store_true", help="Default multi-cams setting")
    parser.add_argument("--acescg", action="store_true", help="Default acescg setting")
    parser.add_argument("--skip-houdini", action="store_true", help="Default skip-houdini setting")
    parser.add_argument("--loop", action="store_true", help="Default loop detection setting")
    parser.add_argument("--loop_period", type=int, help="Default loop detection period")
    parser.add_argument("--loop_num_images", type=int, help="Default loop detection number of images")
    parser.add_argument("--vocab_tree_path", help="Default vocabulary tree path")
    parser.add_argument("--focal_length_mm", type=float, default=None, help="Lens focal length in mm (e.g. 24)")
    parser.add_argument("--sensor_width_mm", type=float, default=None, help="Sensor width in mm (default: 36.0 full-frame)")
    parser.add_argument("--crop", action="store_true", help="Keep original canvas size during undistortion")
    
    args = parser.parse_args()

    target_path = os.path.abspath(args.target_path)
    
    if not os.path.exists(target_path):
        print(f"Error: Path '{target_path}' does not exist.")
        return

    # Check for INI configuration file
    # Using 'global' as the default section name
    config_path = os.path.join(target_path, "batch_config.ini")
    config = configparser.ConfigParser(default_section='global')
    config.optionxform = str # Preserve case sensitivity
    
    if os.path.exists(config_path):
        print(f"[INFO] Found configuration file: {config_path}")
        config.read(config_path)
    else:
        print("[INFO] No batch_config.ini found. Using CLI arguments and defaults.")

    # List all items in the target directory
    items = os.listdir(target_path)
    
    for item in items:
        # Skip hidden directories/files and output directories
        if item.startswith('.') or item.endswith('-output'):
            continue
            
        full_item_path = os.path.join(target_path, item)
        
        if os.path.isdir(full_item_path):
            folder_name = item
            output_path = os.path.join(target_path, f"{folder_name}-output")
            
            print(f"--------------------------------------------------")
            print(f"Processing: {folder_name}")
            
            # Locate run_autotracker.py relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            run_autotracker_script = os.path.join(script_dir, "run_autotracker.py")

            # Base command
            cmd = [sys.executable, run_autotracker_script, full_item_path, output_path]
            
            # Determine effective settings
            # Priority: Folder Section > Global Section > CLI Args
            
            # Helper to resolve value
            def get_setting(ini_key, cli_val, is_bool=False):
                # 1. Check INI (Folder specific or Global via default_section)
                if folder_name in config and ini_key in config[folder_name]:
                    if is_bool: return config.getboolean(folder_name, ini_key)
                    return config.get(folder_name, ini_key)
                if ini_key in config.defaults(): # 'global' section
                    if is_bool: return config.getboolean('global', ini_key)
                    return config.get('global', ini_key)
                # 2. Return CLI val (which might be None or default)
                return cli_val

            # Map settings to command
            s_scale = get_setting('scale', args.scale)
            if s_scale is not None: cmd.extend(['--scale', str(s_scale)])

            s_overlap = get_setting('overlap', args.overlap)
            if s_overlap is not None: cmd.extend(['--overlap', str(s_overlap)])

            s_cam = get_setting('camera_model', args.camera_model)
            if s_cam: cmd.extend(['--camera_model', s_cam])

            s_mask = get_setting('mask', args.mask)
            if s_mask: cmd.extend(['--mask', s_mask])

            s_lut = get_setting('lut', args.lut)
            if s_lut: cmd.extend(['--lut', s_lut])

            s_hfs = get_setting('hfs', args.hfs)
            if s_hfs: cmd.extend(['--hfs', s_hfs])

            if get_setting('multi_cams', args.multi_cams, is_bool=True):
                cmd.append('--multi-cams')

            if get_setting('acescg', args.acescg, is_bool=True):
                cmd.append('--acescg')

            if get_setting('skip_houdini', args.skip_houdini, is_bool=True):
                cmd.append('--skip-houdini')

            if get_setting('loop', args.loop, is_bool=True):
                cmd.append('--loop')
                s_loop_p = get_setting('loop_period', args.loop_period)
                if s_loop_p is not None: cmd.extend(['--loop_period', str(s_loop_p)])
                s_loop_n = get_setting('loop_num_images', args.loop_num_images)
                if s_loop_n is not None: cmd.extend(['--loop_num_images', str(s_loop_n)])
                s_vocab = get_setting('vocab_tree_path', args.vocab_tree_path)
                if s_vocab: cmd.extend(['--vocab_tree_path', s_vocab])

            s_focal = get_setting('focal_length_mm', args.focal_length_mm)
            if s_focal is not None:
                cmd.extend(['--focal_length_mm', str(s_focal)])

            s_sensor = get_setting('sensor_width_mm', args.sensor_width_mm)
            if s_sensor is not None:
                cmd.extend(['--sensor_width_mm', str(s_sensor)])

            if get_setting('crop', args.crop, is_bool=True):
                cmd.append('--crop')

            # Handle dynamic extra arguments (fe.*, sm.*, ma.*)
            def collect_prefixed_settings(prefix):
                settings = {}
                # 1. Start with Global settings
                for key, val in config.defaults().items():
                    if key.startswith(prefix):
                        settings[key[len(prefix):]] = val
                # 2. Overlay Folder settings
                if folder_name in config:
                    for key, val in config[folder_name].items():
                        if key.startswith(prefix):
                            settings[key[len(prefix):]] = val
                return settings

            extra_fe_dict = collect_prefixed_settings("fe.")
            if extra_fe_dict: cmd.extend(["--extra_fe", json.dumps(extra_fe_dict)])
            
            extra_sm_dict = collect_prefixed_settings("sm.")
            if extra_sm_dict: cmd.extend(["--extra_sm", json.dumps(extra_sm_dict)])
            
            extra_ma_dict = collect_prefixed_settings("ma.")
            if extra_ma_dict: cmd.extend(["--extra_ma", json.dumps(extra_ma_dict)])

            print(f"Command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] processing {folder_name}: {e}")
            print(f"--------------------------------------------------\n")

if __name__ == "__main__":
    main()
