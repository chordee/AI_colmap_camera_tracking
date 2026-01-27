import hou
import json
import os
import re
import sys

def create_animated_camera(json_path, global_scale=1, cam_name="Nerfstudio_Animated_Cam", aperture_width=36.0):
    # 1. Check file
    if not os.path.exists(json_path):
        hou.ui.displayMessage(f"Error: File not found at:\n{json_path}")
        return

    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Construct path to background image sequence from json path
    json_dir = os.path.dirname(json_path)
    images_undistorted_dir = os.path.abspath(os.path.join(json_dir, "images_undistorted"))
    # Houdini uses forward slashes
    background_image_path = os.path.join(images_undistorted_dir, "frame_$F6.jpg").replace(os.sep, '/')

    # 2. Get basic information
    frames = data.get("frames", [])
    
    # Sort by number in filename (ensure correct animation order)
    def get_frame_num(frame_data):
        fname = os.path.basename(frame_data['file_path'])
        match = re.search(r'(\d+)', fname)
        return int(match.group(1)) if match else 0
    
    frames.sort(key=get_frame_num)

    if not frames:
        print("No frames found in JSON.")
        return

    # Read resolution and focal length
    img_w = float(data.get("w", 1920))
    img_h = float(data.get("h", 1080))
    fl_x = float(data.get("fl_x", 1000)) # Focal Length in Pixels

    # Convert to Houdini Focal Length (mm)
    focal_mm = (fl_x / img_w) * aperture_width

    # 3. Create Houdini nodes
    obj = hou.node("/obj")
    subnet = obj.node("NeRF_Import")
    if not subnet:
        subnet = obj.createNode("subnet", "NeRF_Import")
    
    # Create camera (destroy and recreate if it already exists)
    cam = subnet.node(cam_name)
    if cam:
        cam.destroy()
    cam = subnet.createNode("cam", cam_name)

    print(f"Creating animation for {len(frames)} frames...")

    # Set static camera parameters
    cam.parm("resx").set(img_w)
    cam.parm("resy").set(img_h)
    cam.parm("aperture").set(aperture_width)
    cam.parm("focal").set(focal_mm)
    cam.parm("iconscale").set(0.5)

    # Set background image for viewport
    cam.parm("vm_background").set(background_image_path)

    # 4. Prepare coordinate transformation matrix (Z-up -> Y-up) (Maybe...)
    correction_rot = hou.hmath.buildRotate(0, 0, 0)

    # 5. Process animation keyframes
    with hou.undos.group("Import Nerfstudio Camera"):
        
        for frame_data in frames:
            # Get Frame Number
            f_num = get_frame_num(frame_data)
            
            # Read matrix
            raw_mtx = frame_data["transform_matrix"]
            
            # [Correction]: Variable name typo fixed, now using raw_mtx
            if isinstance(raw_mtx[0], list):
                flat_mtx = [item for sublist in raw_mtx for item in sublist]
            else:
                flat_mtx = raw_mtx
            
            # Convert to Houdini Matrix4
            h_mtx = hou.Matrix4(tuple(flat_mtx))
            
            # Transpose matrix (Column-Major -> Row-Major)
            h_mtx = h_mtx.transposed()
            
            # Apply coordinate correction
            final_mtx = h_mtx * correction_rot
            
            # Extract transform data
            tra = final_mtx.extractTranslates()
            rot = final_mtx.extractRotates()

            # Prepare values (apply scaling)
            tx = tra[0] * global_scale
            ty = tra[1] * global_scale
            tz = tra[2] * global_scale
            rx, ry, rz = rot

            # Set Keyframes
            target_parms = ["tx", "ty", "tz", "rx", "ry", "rz"]
            values = [tx, ty, tz, rx, ry, rz]

            for p_name, val in zip(target_parms, values):
                k = hou.Keyframe()
                k.setFrame(f_num)
                k.setValue(val)
                k.setExpression("linear()") 
                
                cam.parm(p_name).setKeyframe(k)

    # 6. Set scene range
    start_frame = get_frame_num(frames[0])
    end_frame = get_frame_num(frames[-1])
    
    hou.playbar.setFrameRange(start_frame, end_frame)
    hou.playbar.setPlaybackRange(start_frame, end_frame)
    hou.setFrame(start_frame)

    subnet.layoutChildren()
    cam.parm('vm_bgenable').set(1)
    cam.parm('vm_background').set(background_image_path)
    cam.setInput(0, cam.parent().indirectInputs()[0])
    print(f"Success! Animated camera created at: {cam.path()}")
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        create_animated_camera(json_path=sys.argv[1])
    else:
        print("Usage: hython build_houdinI_scene.py <json_path>")
        
    
    scene = hou.node("/obj").createNode('geo', 'Scene')
    file_node = scene.createNode('file', 'Import_Point_Cloud')
    file_node.parm('file').set(sys.argv[2])
    hou.hipFile.save(sys.argv[3])
