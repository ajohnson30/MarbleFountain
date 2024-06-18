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

from defs import *
from shared import *
from openScadGenerators import *

# Save screw base
outputAssembly = generateCenterScrewRotatingPart()
outputAssembly.save_as_scad(WORKING_DIR + "`Screw.scad")

outputAssembly = cylinder(30, 5, 2)
outputAssembly.save_as_scad(WORKING_DIR + "`Foot.scad")


# Load path data
pathList = pkl.load(open(WORKING_DIR+'path.pkl', 'rb'))
rotList = [calculatePathRotations(path) for path in pathList]

outputAssembly = sphere(0)

# # Generate actual path geometry
for path, rot in zip(pathList, rotList):
    # outputAssembly += generateTrackFromPath(path, rot)
    # outputAssembly += generateTrackFromPath(path[:, :], rot[:, :])
    outputAssembly += generateTrackFromPathSubdiv(path[:, :], rot[:, :])

# Get list of all points which require support
supportAnchors = [calculateSupportAnchorsForPath(path, rot) for path, rot in zip(pathList, rotList)]
supportPoints = np.concatenate(supportAnchors, axis=1)

# Get list of all no-go points
noGoPoints = np.concatenate([path for path in pathList], axis=1) # Subdivide to get intermediate points
noGoPoints -= MARBLE_RAD

# noGoPoints = np.concatenate([subdividePath(path) for path in pathList], axis=1) # Subdivide to get intermediate points
# Calculate lift exclusion zone
liftNoGoRad = SCREW_RAD - POS_DIFF_MIN
liftNoGoZ = np.arange(BASE_OF_MODEL, SIZE_Z, 0.8)
centerPoints = np.array([liftNoGoZ, liftNoGoZ, liftNoGoZ])
centerPoints[0, :] = SIZE_X/2 + liftNoGoRad*np.cos(liftNoGoZ)
centerPoints[1, :] = SIZE_Y/2 + liftNoGoRad*np.sin(liftNoGoZ)

noGoPoints = np.concatenate([noGoPoints, centerPoints], axis=1)


# # Generate supports
# visPath = None
# if SUPPORT_VIS: visPath=WORKING_DIR+'vis/'
# supportColumns = calculateSupports(supportPoints, noGoPoints, visPath)
# supportGeometry = generateSupports(supportColumns)



# Generate base plate connections

# Visualize all support and no-go points
if False:
	for pt in np.swapaxes(supportPoints, 0, 1):
		outputAssembly += sphere(2).translate(pt)

# Add path of marble in 3d to check for intersections
marblePathGeometry = sphere(0)
for fooPath in pathList: marblePathGeometry += getShapePathSet(fooPath, np.zeros_like(fooPath), sphere(MARBLE_RAD/2))
if False:
	outputAssembly += marblePathGeometry

# Add path sections to support screw lift
screwLoadAssembly = sphere(0)
for pathIdx in range(PATH_COUNT):
	angle = np.pi*2*pathIdx/PATH_COUNT
	screwLoadAssembly += generateScrewPathJoins(angle)
screwLoadAssembly = screwLoadAssembly.translate(SCREW_POS)

# Show screw and marble for example
#.translateZ(-(MARBLE_RAD+TRACK_RAD) + BASE_OF_MODEL)
# screwLoadAssembly += sphere(MARBLE_RAD, _fn=40).translateX(SCREW_RAD)
rotatingScrew = generateCenterScrewRotatingPart().translate(SCREW_POS)
os.makedirs(WORKING_DIR+'test/', exist_ok=True)

# (screwLoadAssembly + outputAssembly + supportGeometry).save_as_scad(WORKING_DIR + "MarbleRun.scad")

# (screwLoadAssembly + outputAssembly + supportGeometry + rotatingScrew).save_as_scad(WORKING_DIR + "test/AllComponentsTogether.scad")
# ((supportGeometry) & marblePathGeometry).save_as_scad(WORKING_DIR + "test/supportIntersection.scad")
# (supportGeometry).save_as_scad(WORKING_DIR + "test/supports.scad")
(outputAssembly).save_as_scad(WORKING_DIR + "test/tracks.scad")
