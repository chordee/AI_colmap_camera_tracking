import os
import subprocess
import sys
import argparse
import configparser

def main():
    parser = argparse.ArgumentParser(description="Batch run autotracker on subdirectories of a target path.")
    parser.add_argument("target_path", nargs="?", default=".", help="The path to scan for directories (default: current directory)")
    args = parser.parse_args()

    target_path = os.path.abspath(args.target_path)
    
    if not os.path.exists(target_path):
        print(f"Error: Path '{target_path}' does not exist.")
        return

    # Check for INI configuration file
    config_path = os.path.join(target_path, "batch_config.ini")
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        print(f"[INFO] Found configuration file: {config_path}")
        config.read(config_path)
    else:
        print("[INFO] No batch_config.ini found. Using defaults.")

    # List all items in the target directory
    items = os.listdir(target_path)
    
    for item in items:
        # Skip hidden directories/files
        if item.startswith('.'):
            continue
            
        full_item_path = os.path.join(target_path, item)
        
        if os.path.isdir(full_item_path):
            folder_name = item
            # Create output path relative to the target directory, not the current working directory
            output_path = os.path.join(target_path, f"{folder_name}-output")
            
            print(f"--------------------------------------------------")
            print(f"Processing: {folder_name}")
            print(f"Input: {full_item_path}")
            print(f"Output: {output_path}")
            
            # Base command
            cmd = [sys.executable, "run_autotracker.py", full_item_path, output_path]
            
            # Check if this folder has specific config
            if folder_name in config:
                print(f"[INFO] Applying config settings for [{folder_name}]")
                section = config[folder_name]
                
                # Helper to add optional arguments
                def add_arg(ini_key, cli_flag):
                    if ini_key in section:
                        val = section[ini_key]
                        if val: # Only add if not empty
                            cmd.extend([cli_flag, val])
                            print(f"    + {cli_flag} {val}")

                def add_bool(ini_key, cli_flag):
                    if section.getboolean(ini_key, fallback=False):
                        cmd.append(cli_flag)
                        print(f"    + {cli_flag}")

                # Map configuration to CLI arguments
                add_arg('scale', '--scale')
                add_arg('overlap', '--overlap')
                add_arg('mapper', '--mapper')
                add_arg('camera_model', '--camera_model')
                add_arg('mask', '--mask')
                add_arg('lut', '--lut')
                add_arg('hfs', '--hfs')
                
                add_bool('multi_cams', '--multi-cams')
                add_bool('acescg', '--acescg')
                add_bool('skip_houdini', '--skip-houdini')
            else:
                print(f"[INFO] No specific config found for [{folder_name}], using defaults.")

            print(f"Running command...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] processing {folder_name}: {e}")
            print(f"--------------------------------------------------\n")

if __name__ == "__main__":
    main()
