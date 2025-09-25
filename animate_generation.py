#! /usr/bin/env python

from solid2.extensions.bosl2 import circle, cuboid, sphere, cylinder, \
									heightfield, diff, tag, attach, \
									TOP, BOTTOM, CTR, metric_screws, rect, glued_circles, \
									chain_hull, conv_hull, hull, cube, union, trapezoid, teardrop, skin, sweep
from solid2.core import linear_extrude

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


    geometry, supportAnchors = generateScrewPathJoins(0.0)
    # exit()

    # Add path sections to support screw lift
    screwLoadAssembly = sphere(0)
    screwLoadSupportAnchors = []
    for pathIdx in range(PATH_COUNT):
        angle = getPathAnchorAngle(pathIdx)
        geometry, supportAnchors = generateScrewPathJoins(angle)
        screwLoadAssembly += geometry
        screwLoadSupportAnchors.append(supportAnchors+SCREW_POS[:, None])
    screwLoadAssembly = screwLoadAssembly.translate(SCREW_POS)

    # Get list of all points which require support
    # supportAnchors = [calculateSupportAnchorsForPath(path[:, ::2], rot[:, ::2]) for path, rot in zip(pathList, rotList)]

    supportPoints = np.concatenate([*pathSupportPoints, *screwLoadSupportAnchors], axis=1)

    # Get list of all no-go points
    noGoPoints = np.concatenate([path for path in pathList], axis=1) # Do not subdivide
    # noGoPoints = np.concatenate([subdividePath(path) for path in pathList], axis=1) # Subdivide to get intermediate points
    noGoPoints[2] -= PATH_REPEL_DROP # Repel away only upwards at 45 degree angle

    # supportAnchorPointsConcat = np.concatenate([anchors for anchors in supportAnchors], axis=1) # get concat supportAnchors

    # Calculate lift exclusion zone
    liftNoGoRad = SCREW_RAD
    liftNoGoZ = np.arange(BASE_OF_MODEL, SIZE_Z, 0.55555)
    centerPoints = np.array([liftNoGoZ, liftNoGoZ, liftNoGoZ])
    centerPoints[0, :] = SIZE_X/2 + liftNoGoRad*np.cos(liftNoGoZ)
    centerPoints[1, :] = SIZE_Y/2 + liftNoGoRad*np.sin(liftNoGoZ)

    liftNoGoPtCnt = 20
    liftTrackNogo = np.zeros((3, liftNoGoPtCnt*PATH_COUNT))
    for pathIdx in range(PATH_COUNT):
        angle = getPathAnchorAngle(pathIdx)
        liftTrackNogo[0, liftNoGoPtCnt*(pathIdx):liftNoGoPtCnt*(pathIdx+1)] = np.cos(angle)*(SCREW_RAD + MARBLE_RAD) + SCREW_POS[0]
        liftTrackNogo[1, liftNoGoPtCnt*(pathIdx):liftNoGoPtCnt*(pathIdx+1)] = np.sin(angle)*(SCREW_RAD + MARBLE_RAD) + SCREW_POS[1]
        liftTrackNogo[2, liftNoGoPtCnt*(pathIdx):liftNoGoPtCnt*(pathIdx+1)] = np.linspace(0, SIZE_Z, liftNoGoPtCnt)

    noGoPoints = np.concatenate([noGoPoints, centerPoints, liftTrackNogo], axis=1)

    # Add path of marble in 3d to check for intersections
    marblePathGeometry = sphere(0)
    for fooPath in pathList: marblePathGeometry += getShapePathSet(fooPath, np.zeros_like(fooPath), sphere(MARBLE_RAD))

    # Generate supports
    if GENERATE_SUPPORTS:
        visPath = None

        supportColumns = calculateSupports(supportPoints, noGoPoints, visPath)

        supportGeometry = generateSupportsV2(supportColumns)
    else:
        supportGeometry = sphere(0)
        # supportGeometry = getShapePathSet(supportPoints, None, sphere(1.5), returnIndividual=True)

    print(f"Saving to {file_path}")
    (screwLoadAssembly + outputAssembly + supportGeometry).save_as_scad(file_path)



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


def convertScadFile(png_path, scad_path):
    cmd =  f"~/install/OpenSCAD.AppImage"
    cmd += f" --enable=fast-csg" 
    cmd += f" -o ./{scad_path}"
    cmd += f" --export-format png"
    cmd += f" --preview=throwntogether"
    cmd += f" {png_path}"
    sp.Popen(cmd, shell=True).wait()
    print(f"Ran {cmd}")


print(f"Initializing output directories")

path_folder = Path('FinalDemo_sparse') / 'PathDump'
# path_folder = Path('FinalDemoV3') / 'PathDump'


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

# Reduce path len for testing
# paths = paths[:100]


# Camera args are translate_x,y,z,rot_x,y,z,dist
camera_args = np.zeros((7, len(paths)))







from concurrent.futures import ProcessPoolExecutor
import multiprocessing

cpu_count = multiprocessing.cpu_count()-1 # subtract one as to leave one core so the computer stays usable
print(f"Starting pool executor on {cpu_count} cores")
with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    print(f"Making SCAD files")
    scad_paths = [output_folder / 'scad' / f"{idx:06d}.scad" for idx in range(len(paths))]
    # results = list(executor.map(makeScadFile, paths, scad_paths))
    
    print(f"Making pngs")
    png_paths = [output_folder / 'png' / f"{idx:06d}.png" for idx in range(len(paths))]
    results = list(executor.map(convertScadFile, scad_paths, png_paths))











# Plot update rate over time
if False:
    diffs = np.diff(paths, axis=0)
    diffs = np.abs(diffs)
    for i in range(3): diffs = np.average(diffs, axis=1)
    plt.plot(diffs)
    plt.show()
