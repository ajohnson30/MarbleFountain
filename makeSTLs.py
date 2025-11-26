from os import listdir
import subprocess as sp
from subprocess import call
from stl import mesh
import time

from defs import *


files = listdir(WORKING_DIR)
tasks = []
for f in files:
    if f.find(".scad") >= 0:            # get all .scad files in directory)
        of = f.replace('.scad', '.stl') # name of the outfile .stl
        # cmd = f"openscad -o ./models/{of} --export-format binstl ./output/{f}"

        # if 'Screw' not in of: continue

        # *** Execute windows OpenSCAD here - if multi-threading option found, please add
        cmd = f"/progra~1/OpenSCAD/OpenSCAD  -o ./{WORKING_DIR}/fast_{of} --export-format binstl ./{WORKING_DIR}/{f}"
        
        # cmd = f"~/install/OpenSCAD.AppImage --enable=fast-csg  -o ./{WORKING_DIR}/fast_{of} --export-format binstl ./{WORKING_DIR}/{f}"
        # cmd = f"~/install/OpenSCAD.AppImage -o ./{WORKING_DIR}/fast_{of} --export-format binstl ./{WORKING_DIR}/{f}"
        print(cmd)
        tasks.append(sp.Popen(cmd, shell=True))
        # time.sleep(60*10)

print(f"Start Time: {time.localtime()}")

for foo in tasks:
    print(f"Waiting for {foo}")
    foo.wait()


files = listdir(WORKING_DIR)
for f in files:
    if f.find(".stl") >= 0:
        # Load the STL file
        stl_mesh = mesh.Mesh.from_file(f"{WORKING_DIR}/{f}")
        
        # Get all vertices
        vertices = stl_mesh.vectors.reshape([-1, 3])
        
        # Calculate min and max for each dimension
        min_x, min_y, min_z = np.min(vertices, axis=0)
        max_x, max_y, max_z = np.max(vertices, axis=0)
        
        print(f"\n{f}")
        print(f"   x:{min_x}:{max_x} ({max_x - min_x})")
        print(f"   y:{min_y}:{max_y} ({max_y - min_y})")
        print(f"   z:{min_z}:{max_z} ({max_z - min_z})")


print(f"End Time: {time.localtime()}")
