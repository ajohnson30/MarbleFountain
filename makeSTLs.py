from os import listdir
import subprocess as sp
from subprocess import call
import time

from defs import *

files = listdir(WORKING_DIR)
tasks = []
for f in files:
    if f.find(".scad") >= 0:            # get all .scad files in directory)
        of = f.replace('.scad', '.stl') # name of the outfile .stl
        # cmd = f"openscad -o ./models/{of} --export-format binstl ./output/{f}"

        cmd = f"~/install/OpenSCAD.AppImage -o ./{WORKING_DIR}/fast_{of} --enable=fast-csg --export-format binstl ./{WORKING_DIR}/{f}"
        # cmd = f"~/install/OpenSCAD.AppImage -o ./{WORKING_DIR}/fast_{of} --export-format binstl ./{WORKING_DIR}/{f}"
        
        print(cmd)
        
        tasks.append(sp.Popen(cmd, shell=True))
        # time.sleep(60*10)

for foo in tasks:
    print(f"Waiting for {foo}")
    foo.wait()