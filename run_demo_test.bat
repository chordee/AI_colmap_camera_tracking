:: Default usage: Process video/images, generate sparse point cloud, and build Houdini scene
python run_autotracker.py ./demo-test/walking-forest ./demo-test/walking-forest-output

:: Skip Houdini scene generation if you only need the NeRF/Colmap data:
:: python run_autotracker.py ./demo-test/walking-forest ./demo-test/walking-forest-output --skip-houdini

:: If you need to specify the Houdini install path, use the --hfs flag:
:: python run_autotracker.py ./demo-test/walking-forest ./demo-test/walking-forest-output --hfs "C:/Program Files/Side Effects Software/Houdini 21.0.512"

:: Adjust image scaling (default is 0.5):
:: python run_autotracker.py ./demo-test/walking-forest ./demo-test/walking-forest-output --scale 1.0