import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch run autotracker on subdirectories of a target path.")
    parser.add_argument("target_path", nargs="?", default=".", help="The path to scan for directories (default: current directory)")
    args = parser.parse_args()

    target_path = os.path.abspath(args.target_path)
    
    if not os.path.exists(target_path):
        print(f"Error: Path '{target_path}' does not exist.")
        return

    # List all items in the target directory
    items = os.listdir(target_path)
    
    for item in items:
        # Skip hidden directories/files
        if item.startswith('.'):
            continue
            
        full_item_path = os.path.join(target_path, item)
        
        if os.path.isdir(full_item_path):
            folder_name = item
            output_name = f"{folder_name}-output"
            
            print(f"Running autotracker for: {full_item_path} -> {output_name}")
            
            # Ensure we pass the full path to the folder
            cmd = [sys.executable, "run_autotracker.py", full_item_path, output_name]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {folder_name}: {e}")

if __name__ == "__main__":
    main()
