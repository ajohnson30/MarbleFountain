#! /usr/bin/env python

from solid2.extensions.bosl2 import circle, cuboid, sphere, cylinder, \
									heightfield, diff, tag, attach, \
									TOP, BOTTOM, CTR, metric_screws, rect, glued_circles, \
									chain_hull, conv_hull, hull, cube, union, trapezoid, teardrop, skin, sweep
from solid2.core import linear_extrude, scad_render_to_file

import numpy as np
from copy import deepcopy
import pickle as pkl

from matplotlib import pyplot as plt
import sys
import os
from pathlib import Path

import subprocess as sp

from defs import *
from shared import *
from openScadGenerators import *

def makeScadFile(input_paths, file_path):
    pathList = [subdividePath(path) for path in input_paths]
    rotList = [calculatePathRotations(pathList[pathIdx], getPathAnchorAngle(pathIdx)) for pathIdx in range(len(pathList))]

    outputAssembly = sphere(0)
    pathSupportPoints = []

    # # Generate actual path geometry
    for path, rot in zip(pathList, rotList):
        # outputAssembly += generateTrackFromPath(path, rot)
        # outputAssembly += generateTrackFromPath(path[:, :], rot[:, :])
        tracks, supports = generateTrackFromPathSubdiv(path[:, :], rot[:, :])
        outputAssembly += tracks
        pathSupportPoints += [np.swapaxes(supports, 0, 1)]

    (outputAssembly).save_as_scad(file_path)
    print(f"Saved to {file_path}")
    return

# OpenSCAD camera args for reference
#   --camera arg                      camera parameters when exporting png: 
#                                     =translate_x,y,z,rot_x,y,z,dist or 
#                                     =eye_x,y,z,center_x,y,z
#   --autocenter                      adjust camera to look at object's center
#   --viewall                         adjust camera to fit object
#   --imgsize arg                     =width,height of exported png
#   --render arg                      for full geometry evaluation when exporting
#                                     png
#   --preview arg                     [=throwntogether] -for ThrownTogether 
#                                     preview png
#   --animate arg                     export N animated frames
#   --view arg                        =view options: axes | crosshairs | edges | 
#                                     scales | wireframe
#   --projection arg                  =(o)rtho or (p)erspective when exporting 
#                                     png
#   --csglimit arg                    =n -stop rendering at n CSG elements when 
#                                     exporting png

def convertScadFile(png_path, scad_path, camera_args):
    cmd =  f"~/install/OpenSCAD.AppImage"
    cmd += f" --enable=fast-csg" 
    cmd += f" -o ./{png_path}"
    cmd += f" --export-format png"
    cmd += f" --colorscheme DeepOcean" # I like this one on vibes
    cmd += f" --camera {camera_args}"
    # cmd += f" --view wireframe"
    cmd += f" --imgsize=1920,1080"
    cmd += f" {scad_path}"
    sp.Popen(cmd, shell=True, stdout=open("/dev/null", 'w'), stderr=open("/dev/null", 'w')).wait()
    print(f"Ran {cmd}")



def runBoth(input_path, scad_path, png_path, camera_args):
    makeScadFile(input_path, scad_path)
    convertScadFile(png_path, scad_path, camera_args)

print(f"Initializing output directories")

path_folder = Path('FinalDemo') / 'PathDump'

if False:
    path_folder = Path('FinalDemo_8_track_v2') / 'PathDump'
    path_inflections = [
        0,
        474,
        1068,
        1774,
        2369,
        2937,
        4170,
        5404,
        # 6249,
        6295,
    ]

    path_inflections = [
        0,
        474,
        1068,
        1774,
        2369,
        2937,
        3600,
        4170,
        4800,
        5404,
        6000,
        # 6249,
        6295,
    ]

# Pure inflection points 474,1068,1774,2369,2937,4170,5404,
# Start of real convergence 6249

output_folder = Path('vis_3d')

os.makedirs(output_folder / 'scad', exist_ok=True)
os.makedirs(output_folder / 'png', exist_ok=True)

# Load path data
print(f"Loading path data")
filecount = len(os.listdir(path_folder))
paths = []
for fileIdx in range(filecount):
    paths.append(pkl.load(open(path_folder / (str(fileIdx).rjust(6, '0')+'.pkl'), 'rb')))

paths = np.array(paths)

# Paths indexing is (iteration #, path #, XYZ, point #)

path_centers = [np.average(paths[:, :, i, :]) for i in range(3)]

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def rolling_average_axis0(arr, window_size):
    # Create sliding windows along axis 0
    windowed = sliding_window_view(arr, window_shape=window_size, axis=0)
    # Compute mean along the window dimension (last axis)
    return np.mean(windowed, axis=-1)

if False: # Downsample paths
    paths = np.concatenate([
        paths[:100],
        paths[100:200][::5],
        paths[200::20],
        ])

rolling_avg = rolling_average_axis0(paths, window_size=30)
paths = rolling_avg

if False: # Resample to fixed frame count
    frame_count = 8*30
    new_path_pts = np.zeros((frame_count, paths.shape[1], paths.shape[2], paths.shape[3]))
    for pathIdx in range(paths.shape[1]):
        for axisIdx in range(paths.shape[2]):
            for ptIdx in range(paths.shape[3]):
                new_path_pts[:, pathIdx, axisIdx, ptIdx] = np.interp(frame_count, np.linspace(0.0, 1.0, len(paths[:, pathIdx, axisIdx, ptIdx])), paths[:, pathIdx, axisIdx, ptIdx])
    paths = new_path_pts


# Plot update rate over time
if False:
    print(paths.shape)
    diffs = np.diff(paths, axis=0)
    diffs = np.abs(diffs)
    for i in range(3):
        diffs = np.average(diffs, axis=1)
        print(diffs.shape)
    plt.style.use('dark_background')

    if True:
        plt.plot(diffs)
        plt.show()
        exit()
    else:
        plt.plot(np.linspace(0, 1.0, len(diffs)), diffs)
        plt.vlines(np.array(path_inflections) / path_inflections[-1], 0, 1)

# Interpolate to run on beat
frame_rate = 30
if False:
    step_duration = 0.78125 * 2
    # step_duration = 0.5
    frame_count = int(frame_rate*step_duration * (len(path_inflections) - 1))

    print(f"frame_count:{frame_count}")
    print(f"run_duration S:{frame_count/frame_rate}")

    frame_indices = np.arange(frame_count)
    inflection_frame_indices = np.linspace(0, path_inflections[-1], paths.shape[0])

    # Get interpolation pairs to equalize input
    frame_iteration_sample_points = np.interp(
        inflection_frame_indices,
        path_inflections,
        np.linspace(0.0, frame_count, len(path_inflections)),
    )

    new_path_pts = np.zeros((frame_count, paths.shape[1], paths.shape[2], paths.shape[3]))
    for pathIdx in range(paths.shape[1]):
        for axisIdx in range(paths.shape[2]):
            for ptIdx in range(paths.shape[3]):
                new_path_pts[:, pathIdx, axisIdx, ptIdx] = np.interp(frame_indices, frame_iteration_sample_points, paths[:, pathIdx, axisIdx, ptIdx])

    paths = new_path_pts

# Plot update rate over time
if False:
    print(paths.shape)
    diffs = np.diff(paths, axis=0)
    diffs = np.abs(diffs)
    for i in range(3):
        diffs = np.average(diffs, axis=1)
        print(diffs.shape)
    # plt.plot(np.linspace(0, 1.0, len(diffs)), diffs)
    plt.plot(diffs)
    plt.show()
    exit()

# Reduce path len for testing
# paths = paths[::2]



# Camera args are translate_x,y,z,rot_x,y,z,dist
camera_args = np.zeros((7, len(paths)))

# Aim for center of paths consistently across all schots
for i in range(3): camera_args[i] = path_centers[i]
camera_args[6] = 900.0

# Reasonable fixed angle
# camera_args[3] = 75

# Camera go spinny weeee
# camera_args[3] = np.linspace(0.0, 70.0, len(paths))
camera_args[3] = np.interp(
    np.linspace(0.0, 1.0, len(paths)),
    [0.0, 1.0],
    [30.0, 80.0]
)
camera_args[5] = np.linspace(0.0, 360.0*3.0, len(paths))

# # Noise jumps at 836 & 1734
# camera_args[3] = np.interp(
#     np.arange(len(paths)),
#     [0.0, 836, len(paths)],
#     [0.0, 60.0, 70.0]
# )

# camera_args[5] = np.interp(
#     np.arange(len(paths)),
#     [0.0, 836, len(paths)],
#     [0.0, 360.0*1.5, 360.0*3]
# )

# Only print final frame
if False:
    paths = paths[-1:]
    camera_args = camera_args[:, -1:]

# make camera args into csv separated string
camera_args = [",".join([str(bar) for bar in foo]) for foo in camera_args.swapaxes(0,1)]
for foo in camera_args: print(foo)

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

cpu_count = multiprocessing.cpu_count()-1 # subtract one as to leave one core so the computer stays usable
print(f"Starting pool executor on {cpu_count} cores")
with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    print(f"Making SCAD files")
    scad_paths = [output_folder / 'scad' / f"{idx:06d}.scad" for idx in range(len(paths))]
    png_paths = [output_folder / 'png' / f"{idx:06d}.png" for idx in range(len(paths))]
    results = list(executor.map(runBoth, paths, scad_paths, png_paths, camera_args))

print("Joining into video")
compile_vid_command = f"ffmpeg -framerate {frame_rate} -pattern_type glob -i \"{str(output_folder / 'png' / '*.png')}\" -c:v libx264 -crf 18 -pix_fmt yuv420p {str(output_folder/"animation.mp4")}"
# ffmpeg -framerate 30 -pattern_type glob -i "vis/*.png" -c:v libx264 -crf 18 -pix_fmt yuv420p "animation.mp4"
sp.Popen(compile_vid_command, shell=True).wait()
print(f"Finished, vid build with {compile_vid_command}")

# ffmpeg -framerate 30 -pattern_type glob -i "png/*.png" -c:v libx264 -crf 18 -pix_fmt yuv420p "test_animation.mp4"

# Example renders for each color scheme
if False:
    for foo in ["Cornfield","Metallic","Sunset","Starnight","BeforeDawn","Nature","DeepOcean","Solarized","Tomorrow","Tomorrow","Night","ClearSky","Monotone"]:
        cmd =  f"~/install/OpenSCAD.AppImage"
        cmd += f" --enable=fast-csg" 
        cmd += f" -o ./{output_folder / (foo+'.png')}"
        cmd += f" --colorscheme {foo}"
        cmd += f" --export-format png"
        cmd += f" {output_folder / "scad" / "000000.scad"}"
        sp.Popen(cmd, shell=True).wait()
