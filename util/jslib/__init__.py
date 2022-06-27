from pathlib import Path
from os.path import dirname, join
JQUERY = Path(join(dirname(__file__), 'jquery.js')).read_text()
THREEJS = Path(join(dirname(__file__), 'threejs.js')).read_text()
ORBIT_CTR = Path(join(dirname(__file__), 'orbit_control.js')).read_text()
VIDEO_VIEWER = Path(join(dirname(__file__), 'video_viewer.js')).read_text()
__all__ = ["JQUERY", "ORBIT_CTR", "THREEJS", "VIDEO_VIEWER"]
